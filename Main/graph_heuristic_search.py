#!/usr/bin/env python3
"""
Graph Heuristic Search implementation following Algorithm 2 from RoboGrammar paper.
Core algorithm implementation with Graph Neural Network for design value prediction.

A search state is a partial hand design (palm + five finger slots). A rollout assigns
servo count / grammar counts / fingertip type finger by finger, choosing among the legal
successors with the value network as the heuristic plus exploration noise:
  * selection="gumbel" (paper, Algorithm 1 line 8): argmax_o V(o) + tau_k * Gumbel. The predicted
    values are standardized within each decision so tau is dimensionless, and tau_k is annealed
    over iterations (1.0 -> 0.05 recommended), i.e. Boltzmann sampling from broad to near-greedy.
  * selection="epsilon": epsilon-greedy with the same annealed schedule.
Every partial state a rollout visits is kept and, once the completed hand has been evaluated, is
trained toward the best reward seen among its descendants (RoboGrammar's V* target).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import json
import math
import os
import statistics
import time
from hand_groups import HandGroup
import copy

# Fingertip vocabulary (fixed order for one-hot encoding)
FINGERTIP_CHOICES = ["standard", "thinner", "wedged"]
TIP2IDX = {name: i for i, name in enumerate(FINGERTIP_CHOICES)}
GROUP_NAMES = [g.value for g in HandGroup]  # ["sym3", "sym4", "sym5", "anth21", "anth27", "anth33"]

# Finger fields the search assigns -> node key recording whether the field has been assigned yet.
# Unassigned fields are masked out of the GNN input so placeholder values never leak into scores.
ASSIGNED_FLAGS = {
    "servo_count": "servo_assigned",
    "grammar_1_count": "g1_assigned",
    "grammar_2_count": "g2_assigned",
    "fingertip_type": "tip_assigned",
}

# 6 scalars + 4 servo one-hot + 6 slot one-hot + 3 tip one-hot + 6 group one-hot + 4 assigned flags
NODE_FEATURE_DIM = 6 + 4 + 6 + len(FINGERTIP_CHOICES) + len(GROUP_NAMES) + len(ASSIGNED_FLAGS)

PARTIAL_KEY_PREFIX = "partial:"   # lookup-table namespace for partial search states
FAILED_SCORE = -1000.0            # sentinel the evaluator returns for designs that could not be built/evaluated


class GraphNeuralNetwork(nn.Module):
    """Simple GNN for design value estimation"""

    def __init__(self, node_features: int = NODE_FEATURE_DIM, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Node embedding layers
        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                self.dropout,
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])

        # Global pooling and output
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_features]
            edge_indices: [2, num_edges] - source and target node indices
        Returns:
            Scalar value prediction for the graph
        """
        h = self.node_embed(node_features)  # [num_nodes, hidden_dim]

        for layer in self.message_layers:
            messages = []
            for i in range(h.size(0)):
                neighbors = edge_indices[1][edge_indices[0] == i]
                if len(neighbors) > 0:
                    neighbor_features = h[neighbors]
                    node_features_expanded = h[i].unsqueeze(0).expand(len(neighbors), -1)
                    combined = torch.cat([node_features_expanded, neighbor_features], dim=1)
                    message = layer(combined).mean(dim=0)
                else:
                    message = torch.zeros_like(h[i])
                messages.append(message)
            h = torch.stack(messages)

        graph_embedding = h.mean(dim=0)
        value = self.global_pool(graph_embedding)
        return value.squeeze()


@dataclass
class DesignGraph:
    """Represents a hand design as a graph structure"""
    nodes: List[Dict]  # Node features (finger type, servo presence, etc.)
    edges: List[Tuple[int, int]]  # Edge connections
    terminals: Set[int]  # Terminal node indices (fully specified)
    non_terminals: Set[int]  # Non-terminal node indices (need expansion)
    design_string: str  # String representation for hashing/comparison
    group: Optional[str] = None

    def __hash__(self):
        return hash(self.design_string)

    def __eq__(self, other):
        return isinstance(other, DesignGraph) and self.design_string == other.design_string

    def __len__(self):
        return len(self.nodes)

    def is_complete(self) -> bool:
        """Check if design is fully specified (no non-terminals)"""
        return len(self.non_terminals) == 0

    @staticmethod
    def one_hot(i, K):
        v = [0.0] * K
        if 0 <= i < K:
            v[i] = 1.0
        return v

    @staticmethod
    def _assigned_flags(node: Dict) -> List[bool]:
        """Per-field assignment flags; nodes that never went through the search count as fully assigned."""
        return [bool(node.get(flag, True)) for flag in ASSIGNED_FLAGS.values()]

    def to_tensor_data(self, group_name: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensors for GNN with group conditioning. Unassigned fields are zeroed."""
        group_oh = [0.0] * len(GROUP_NAMES)
        if group_name in GROUP_NAMES:
            group_oh[GROUP_NAMES.index(group_name)] = 1.0

        node_tensor = []
        for node in self.nodes:
            fid = int(node.get('finger_id', 0))
            is_base = 1.0 if node.get('is_base', fid == 0) else 0.0
            is_term = 1.0 if node.get('is_terminal', False) else 0.0
            s = int(node.get('servo_count', 0))
            g1 = float(node.get('grammar_1_count', 0))
            g2 = float(node.get('grammar_2_count', 0))
            s_ok, g1_ok, g2_ok, tip_ok = self._assigned_flags(node)

            g1n = (g1 - 1.0) / 9.0 if (g1_ok and g1 > 0) else 0.0
            g2n = (g2 - 1.0) / 9.0 if (g2_ok and g2 > 0) else 0.0

            present = 1.0 if (s > 0 or fid == 0) else 0.0
            s_oh = self.one_hot(s, 4) if s_ok else [0.0] * 4
            pos_oh = self.one_hot(max(0, fid), 6)
            actionable = 1.0 if (present and not node.get('is_terminal', False) and fid > 0) else 0.0

            tip_idx = TIP2IDX.get(str(node.get('fingertip_type', 'standard')), 0)
            tip_oh = self.one_hot(tip_idx, len(TIP2IDX)) if (tip_ok and present) else [0.0] * len(TIP2IDX)

            feats = [
                is_base, is_term, present, actionable, g1n, g2n,   # 6
                *s_oh,                                             # 4
                *pos_oh,                                           # 6
                *tip_oh,                                           # 3
                *group_oh,                                         # 6
                float(s_ok), float(g1_ok), float(g2_ok), float(tip_ok),  # 4
            ]
            assert len(feats) == NODE_FEATURE_DIM
            node_tensor.append(feats)

        node_features = torch.tensor(node_tensor, dtype=torch.float32)

        if self.edges:
            edge_indices = torch.tensor(self.edges, dtype=torch.long).t()
            rev = edge_indices.flip(0)
            edge_indices = torch.cat([edge_indices, rev], dim=1)
        else:
            edge_indices = torch.zeros((2, 0), dtype=torch.long)

        return node_features, edge_indices

    def partial_state_key(self, group_name: Optional[str]) -> str:
        """Content key of a (partial) search state. Unassigned fields are written as '?' so that
        two states that differ only in placeholder values share one key. Lives in its own
        namespace so it can never collide with the effective key of a physical hand."""
        parts = []
        for node in self.nodes[1:]:
            s_ok, g1_ok, g2_ok, tip_ok = self._assigned_flags(node)
            parts.append("%s,%s,%s,%s,%d" % (
                node.get('servo_count', 0) if s_ok else "?",
                node.get('grammar_1_count', 0) if g1_ok else "?",
                node.get('grammar_2_count', 0) if g2_ok else "?",
                node.get('fingertip_type', 'standard') if tip_ok else "?",
                1 if node.get('is_terminal', False) else 0,
            ))
        return f"{PARTIAL_KEY_PREFIX}{group_name}:" + ";".join(parts)


# Attributes carried across canonicalization (search bookkeeping attached to DesignGraph instances)
_CARRIED_ATTRS = ("_effective_key", "_effective_group", "_effective_graph", "_value_key", "evaluated_group", "trajectory")


def canonicalize_fingers(design: "DesignGraph") -> "DesignGraph":
    """
    Produce a safe canonical 6-node layout while PRESERVING finalization:
    - node 0 is palm (terminal, base)
    - exactly 5 finger slots (1..5)
    - active fingers (servo_count >= 2) first, then ghosts (servo_count < 2)
    - edges rebuilt as star: (0,i) for i=1..5
    - *explicit is_terminal stays true* even if servo_count >= 2
    """
    nodes = list(design.nodes) if getattr(design, "nodes", None) else []

    if not nodes:
        nodes = [{'finger_id': 0, 'is_base': True, 'is_terminal': True}]
    else:
        n0 = copy.deepcopy(nodes[0])
        n0['finger_id'] = 0
        n0['is_base'] = True
        n0['is_terminal'] = True
        nodes[0] = n0

    if len(nodes) > 6:
        nodes = nodes[:6]

    while len(nodes) < 6:
        idx = len(nodes)
        nodes.append({
            'finger_id': idx,
            'grammar_1_count': 0,
            'grammar_2_count': 0,
            'servo_count': 0,
            'fingertip_type': 'standard',
            'is_terminal': True,   # ghosts are terminal
            'is_base': False,
        })

    active_idx = [i for i in range(1, 6) if nodes[i].get('servo_count', 0) >= 2]
    ghost_idx = [i for i in range(1, 6) if nodes[i].get('servo_count', 0) < 2]
    ordered = [0] + active_idx + ghost_idx
    idx_map = {old_i: new_fid for new_fid, old_i in enumerate(ordered)}

    new_nodes = []
    for new_fid, old_i in enumerate(ordered):
        n = copy.deepcopy(nodes[old_i])
        n['finger_id'] = new_fid
        if new_fid == 0:
            n['is_base'] = True
            n['is_terminal'] = True
        else:
            servo = int(n.get('servo_count', 0))
            was_terminal = bool(n.get('is_terminal', False))
            n['is_base'] = False
            n['is_terminal'] = was_terminal or (servo < 2)
        new_nodes.append(n)

    new_edges = [(0, i) for i in range(1, 6)]

    orig_terms = set(getattr(design, 'terminals', set()))
    orig_non = set(getattr(design, 'non_terminals', set()))
    new_terminals = {idx_map.get(i, i) for i in orig_terms if idx_map.get(i, i) < 6}
    new_non_terms = {idx_map.get(i, i) for i in orig_non if idx_map.get(i, i) < 6}

    for i in range(1, 6):
        if new_nodes[i].get('servo_count', 0) < 2 or new_nodes[i].get('is_terminal', False):
            new_terminals.add(i)
            new_non_terms.discard(i)

    new_terminals.add(0)
    new_non_terms.discard(0)

    canon = DesignGraph(
        nodes=new_nodes,
        edges=new_edges,
        terminals=new_terminals,
        non_terminals=new_non_terms,
        design_string=f"{getattr(design, 'design_string', 'design')}_canon",
        group=design.group,
    )
    for attr in _CARRIED_ATTRS:
        if hasattr(design, attr):
            setattr(canon, attr, getattr(design, attr))
    return canon


class SequentialFingerSchedule:
    """Finger 1→2→3→4→5 OR 5→4→3→2→1; per finger phases: servo -> (g1 if servo==3) -> g2 -> tip -> finalize."""
    def __init__(self, design: DesignGraph, phases: List[str], *, direction: str = "forward"):
        self.phases = phases
        base_order = [f for f in [1, 2, 3, 4, 5] if f in design.non_terminals]
        if direction == "backward":
            base_order = list(reversed(base_order))
        self.order = base_order
        self.phase_idx = {f: 0 for f in self.order}
        self.f_idx = 0  # which finger in self.order is current

    def current(self) -> Optional[Tuple[int, str]]:
        while self.f_idx < len(self.order) and self.phase_idx[self.order[self.f_idx]] >= len(self.phases):
            self.f_idx += 1
        if self.f_idx >= len(self.order):
            return None
        f = self.order[self.f_idx]
        return f, self.phases[self.phase_idx[f]]

    def advance_for(self, finger: int, auto_steps: int = 1):
        if finger in self.phase_idx:
            self.phase_idx[finger] = min(self.phase_idx[finger] + auto_steps, len(self.phases))

    def done(self) -> bool:
        return self.f_idx >= len(self.order)


class HandDesignGenerator:
    """Generates hand designs following grammar rules"""

    PHASES = ["set_servo_count", "set_grammar_1", "set_grammar_2", "set_fingertip_type", "finalize_finger"]

    def __init__(self, max_fingers: int = 5, max_servos_per_finger: int = 4):
        self.max_fingers = max_fingers
        self.max_servos_per_finger = max_servos_per_finger

        self.grammar_rules = {
            'set_grammar_1': self._set_grammar_1_rule,
            'set_grammar_2': self._set_grammar_2_rule,
            'set_servo_count': self._set_servo_count_rule,
            'set_fingertip_type': self._set_fingertip_type_rule,
            'finalize_finger': self._finalize_finger_rule
        }

    @staticmethod
    def _choose_finger_idx(design: DesignGraph, params: Optional[Dict], *, min_servo: int = 0) -> Optional[int]:
        candidates = [
            i for i in design.non_terminals
            if i > 0 and int(design.nodes[i].get("servo_count", 0)) >= min_servo
        ]
        if not candidates:
            return None
        if params and "finger_idx" in params:
            return int(params["finger_idx"])
        return random.choice(candidates)

    @staticmethod
    def _clone_with_finger_updates(
        design: DesignGraph,
        finger_idx: int,
        *,
        suffix: str,
        updates: Dict,
        terminals: Optional[Set[int]] = None,
        non_terminals: Optional[Set[int]] = None,
    ) -> DesignGraph:
        new_nodes = design.nodes.copy()
        new_nodes[finger_idx] = new_nodes[finger_idx].copy()
        new_nodes[finger_idx].update(updates)
        return DesignGraph(
            new_nodes,
            design.edges,
            design.terminals if terminals is None else terminals,
            design.non_terminals if non_terminals is None else non_terminals,
            design.design_string + suffix,
            group=design.group,
        )

    def options_for_phase(self, design: DesignGraph, finger_idx: int, phase: str) -> List[Dict]:
        """Small legal option set for this finger+phase, with dependencies enforced."""
        if finger_idx == 0 or finger_idx not in design.non_terminals:
            return []

        is_anth = (design.group in ("anth21", "anth27", "anth33"))
        servo = int(design.nodes[finger_idx].get("servo_count", 0))
        opts: List[Dict] = []

        if phase == "set_servo_count":
            if is_anth and finger_idx == 1:
                opts = [{'finger_idx': finger_idx, 'servo_count': 3}]
            else:
                opts = [{'finger_idx': finger_idx, 'servo_count': s} for s in (2, 3)]

        elif phase == "set_grammar_1":
            if servo == 3:
                opts = [{'finger_idx': finger_idx, 'grammar_1_count': c} for c in range(1, 11)]
            else:
                # Single forced option (0): the rollout auto-applies it.
                opts = [{'finger_idx': finger_idx, 'grammar_1_count': 0}]

        elif phase == "set_grammar_2":
            if servo >= 2:
                opts = [{'finger_idx': finger_idx, 'grammar_2_count': c} for c in range(1, 11)]

        elif phase == "set_fingertip_type":
            opts = [{'finger_idx': finger_idx, 'fingertip_type': tip} for tip in FINGERTIP_CHOICES]

        elif phase == "finalize_finger":
            if servo >= 2:
                opts = [{'finger_idx': finger_idx}]

        return opts

    def get_initial_design(self, group: Optional[HandGroup] = None) -> DesignGraph:
        """Create the root search state for a group: active fingers with every field still
        unassigned (placeholders are masked out of the GNN input), trailing ghost fingers."""
        if group is None:
            active_count = random.randint(3, 5)
        elif group == HandGroup.SYM3:
            active_count = 3
        elif group == HandGroup.SYM4:
            active_count = 4
        elif group == HandGroup.SYM5:
            active_count = 5
        elif group in [HandGroup.ANTH21, HandGroup.ANTH27, HandGroup.ANTH33]:
            active_count = 5
        else:
            active_count = random.randint(3, 5)

        nodes = [{'finger_id': 0, 'is_base': True, 'is_terminal': True}]

        for i in range(1, active_count + 1):
            nodes.append({
                'finger_id': i,
                'grammar_1_count': 0,
                'grammar_2_count': 1,
                'servo_count': 2,          # placeholder: keeps the slot "active" for canonicalization
                'fingertip_type': 'standard',
                'is_terminal': False,
                'is_base': False,
                'servo_assigned': False,
                'g1_assigned': False,
                'g2_assigned': False,
                'tip_assigned': False,
            })
        for i in range(active_count + 1, 6):
            nodes.append({
                'finger_id': i,
                'grammar_1_count': 0,
                'grammar_2_count': 0,
                'servo_count': 0,
                'fingertip_type': 'standard',
                'is_terminal': True,
                'is_base': False,
            })

        edges = [(0, i) for i in range(1, 6)]
        terminals = {0} | {i for i in range(active_count + 1, 6)}
        non_terminals = {i for i in range(1, active_count + 1)}

        gstr = group.value if group else "nogroup"
        return DesignGraph(nodes, edges, terminals, non_terminals, f"base_{active_count}active_{gstr}_canonical", group=gstr)


    def apply_rule(self, design: DesignGraph, rule: str, params: Optional[Dict] = None) -> DesignGraph:
        if rule not in self.grammar_rules:
            raise ValueError(f"Unknown rule: {rule}")
        out = self.grammar_rules[rule](design, params)
        out.group = design.group
        return out

    def _set_grammar_1_rule(self, design: DesignGraph, params: Optional[Dict] = None) -> DesignGraph:
        finger_idx = self._choose_finger_idx(design, params)
        if finger_idx is None:
            return design
        new_count = params.get('grammar_1_count') if params and 'grammar_1_count' in params else random.randint(1, 10)
        return self._clone_with_finger_updates(
            design, finger_idx,
            suffix=f"_g1f{finger_idx}c{new_count}",
            updates={"grammar_1_count": int(new_count), "g1_assigned": True},
        )

    def _set_grammar_2_rule(self, design: DesignGraph, params: Optional[Dict] = None) -> DesignGraph:
        finger_idx = self._choose_finger_idx(design, params)
        if finger_idx is None:
            return design
        new_count = params.get('grammar_2_count') if params and 'grammar_2_count' in params else random.randint(1, 10)
        return self._clone_with_finger_updates(
            design, finger_idx,
            suffix=f"_g2f{finger_idx}c{new_count}",
            updates={"grammar_2_count": int(new_count), "g2_assigned": True},
        )

    def _set_servo_count_rule(self, design: DesignGraph, params: Optional[Dict] = None) -> DesignGraph:
        finger_idx = self._choose_finger_idx(design, params)
        if finger_idx is None:
            return design

        is_anth = (design.group in ("anth21", "anth27", "anth33"))
        if is_anth and finger_idx == 1:
            new_count = 3  # thumb lock
        else:
            proposed = params.get('servo_count') if params and 'servo_count' in params else random.choice([2, 3])
            new_count = proposed if proposed in (2, 3) else 2

        return self._clone_with_finger_updates(
            design, finger_idx,
            suffix=f"_s{finger_idx}c{new_count}",
            updates={"servo_count": int(new_count), "servo_assigned": True},
        )

    def _set_fingertip_type_rule(self, design: DesignGraph, params: Optional[Dict] = None) -> DesignGraph:
        finger_idx = self._choose_finger_idx(design, params)
        if finger_idx is None:
            return design
        tip = params.get('fingertip_type') if params and 'fingertip_type' in params else random.choice(FINGERTIP_CHOICES)
        return self._clone_with_finger_updates(
            design, finger_idx,
            suffix=f"_tipf{finger_idx}{tip}",
            updates={"fingertip_type": tip, "tip_assigned": True},
        )

    def _finalize_finger_rule(self, design: DesignGraph, params: Optional[Dict] = None) -> DesignGraph:
        finger_idx = self._choose_finger_idx(design, params, min_servo=2)
        if finger_idx is None:
            return design
        new_term = design.terminals.copy()
        new_term.add(finger_idx)
        new_non = design.non_terminals.copy()
        new_non.remove(finger_idx)
        return self._clone_with_finger_updates(
            design, finger_idx,
            suffix=f"_fin{finger_idx}",
            updates={"is_terminal": True},
            terminals=new_term,
            non_terminals=new_non,
        )


class GraphHeuristicSearch:
    """Main algorithm implementation following RoboGrammar Algorithm 2"""

    SELECTION_MODES = ("gumbel", "epsilon")

    def __init__(self,
                 num_iterations: int = 100,
                 num_candidates: int = 10,
                 opt_iterations: int = 50,
                 batch_size: int = 8,
                 epsilon: float = 0.1,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 debug: bool = False,
                 selection: str = "gumbel"):

        self.debug = debug
        if selection not in self.SELECTION_MODES:
            raise ValueError(f"selection must be one of {self.SELECTION_MODES}, got {selection!r}")
        self.selection = selection

        self.N = num_iterations
        self.M = num_candidates
        self.K = num_candidates  # Number of designs to sample per iteration
        self.opt_iter = opt_iterations
        self.batch_size = batch_size
        # Exploration parameter, annealed by run() from `epsilon` to `epsilon_final`:
        # Gumbel temperature (selection="gumbel") or epsilon-greedy rate (selection="epsilon").
        self.epsilon = epsilon
        self.epsilon_final = 0.05
        self.device = device
        self.num_iterations = num_iterations
        self.num_candidates = num_candidates
        self.checkpoint_dir = "ghs_checkpoints"

        self.generator = HandDesignGenerator()
        self.value_network = GraphNeuralNetwork().to(device)
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)

        # (key, group) -> best reward seen. Keys are effective keys of physical hands or
        # "partial:" state keys; the two namespaces never collide.
        self.lookup_table: Dict[Tuple[str, str], float] = {}
        self.best_design: Optional[DesignGraph] = None
        self.best_reward: float = 0.0

        # Training set: one DesignGraph per (key, group), tagged with _value_key / evaluated_group
        self.seen_designs: List[DesignGraph] = []
        self._seen_keys: Set[Tuple[str, str]] = set()
        self.num_evaluated: int = 0

        self.evaluator = None

    # ------------------------------------------------------------------ scoring
    def _score_partial(self, design: DesignGraph, group_name: str, cache: Dict[str, float]) -> float:
        key = f"p:{group_name}:{design.design_string}"
        if key in cache:
            return cache[key]
        node_features, edge_indices = design.to_tensor_data(group_name=group_name)
        node_features = node_features.to(self.device)
        edge_indices = edge_indices.to(self.device)
        with torch.no_grad():
            self.value_network.eval()
            v = self.value_network(node_features, edge_indices).item()
        self.value_network.train()
        cache[key] = v
        return v

    # ------------------------------------------------------------------ candidate generation
    def generate_candidate_designs(self) -> List[DesignGraph]:
        """K epsilon-greedy rollouts, alternating finger order, deduplicated by effective (physical) key.
        A duplicate is re-rolled a few times with increasing exploration before being dropped."""
        assert self.evaluator is not None, "Set self.evaluator before generation"
        grp_obj = self.evaluator._current_group()
        unique: List[DesignGraph] = []
        seen_eff: Set[str] = set()
        batch_path_tabu: set = set()

        for k in range(self.K):
            direction = "forward" if (k % 2 == 0) else "backward"
            eps = self.epsilon
            eff_key = None
            cand = None
            for _attempt in range(4):
                cand = self._generate_greedy_design(
                    direction=direction, rng_seed=random.randrange(1_000_000_000),
                    exploration=eps, batch_path_tabu=batch_path_tabu,
                )
                eff_key = self.compute_effective_key_for_group(cand, grp_obj)
                if eff_key not in seen_eff:
                    break
                eps = min(1.0, eps * 2.0 + 0.1)   # duplicate: re-roll with more exploration
            if eff_key in seen_eff:
                if self.debug:
                    print(f"[dedupe] dropped duplicate candidate eff_key={eff_key}")
                continue
            seen_eff.add(eff_key)
            unique.append(cand)

        if self.debug and len(unique) < self.K:
            print(f"[warn] produced {len(unique)} unique candidates (target {self.K})")
        return unique

    @staticmethod
    def _gumbel(rng: random.Random) -> float:
        return -math.log(-math.log(max(rng.random(), 1e-12)))

    def _rank_successors(self, scores: List[float], exploration: float, rng: random.Random) -> List[int]:
        """Successor preference order (best first).

        gumbel : argsort of z_i + tau * g_i with g_i ~ Gumbel(0, 1), where z_i are the predicted values
                 standardized within this decision. This is the paper's rule (Algorithm 1, line 8) and
                 equals sampling successors with probability proportional to exp(z_i / tau); standardizing
                 makes tau independent of the reward scale. exploration == tau.
        epsilon: uniformly random order with probability `exploration`, otherwise by predicted value.
        """
        idx = list(range(len(scores)))
        if self.selection == "epsilon":
            if rng.random() < exploration:
                rng.shuffle(idx)
                return idx
            return sorted(idx, key=lambda i: (-scores[i], rng.random()))

        sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        if sd > 1e-6:
            mu = statistics.fmean(scores)
            z = [(v - mu) / sd for v in scores]
        else:
            z = [0.0] * len(scores)   # no usable preference (e.g. untrained network): pure exploration
        noisy = [z_i + exploration * self._gumbel(rng) for z_i in z]
        return sorted(idx, key=lambda i: -noisy[i])

    def _generate_greedy_design(
        self,
        *,
        direction: str = "forward",
        rng_seed: int = 0,
        exploration: float = 0.0,
        batch_path_tabu: Optional[set] = None,
    ) -> DesignGraph:
        """One rollout from the group's root state to a complete hand, choosing successors with
        `_rank_successors` (Gumbel temperature or epsilon = `exploration`).
        The returned design carries `.trajectory`: the partial states visited on the way."""
        rng = random.Random(rng_seed)   # local RNG: never reseeds the global stream

        grp_obj = self.evaluator._current_group()
        grp = grp_obj.value if grp_obj else "default"

        cur = self.generator.get_initial_design(group=grp_obj)
        cache: Dict[str, float] = {}
        applied: Set[Tuple[int, str]] = set()
        trajectory: List[DesignGraph] = []
        path_key: list = []

        sched = SequentialFingerSchedule(cur, self.generator.PHASES, direction=direction)
        MAX_STEPS = 60
        step = 0

        while step < MAX_STEPS and not sched.done() and not cur.is_complete():
            item = sched.current()
            if item is None:
                break
            finger, phase = item

            if (finger, phase) in applied:
                sched.advance_for(finger)
                continue

            options = self.generator.options_for_phase(cur, finger, phase)
            if not options:
                sched.advance_for(finger)
                continue

            if len(options) == 1:
                # Forced choice (anthro thumb servo, grammar_1 without a base servo, finalize): no decision.
                cur = self.generator.apply_rule(cur, phase, {**options[0], 'finger_idx': finger})
                applied.add((finger, phase))
                sched.advance_for(finger)
                trajectory.append(cur)
                step += 1
                continue

            succs, succ_keys = [], []
            for opt in options:
                succs.append(self.generator.apply_rule(cur, phase, {**opt, 'finger_idx': finger}))
                succ_keys.append((finger, phase) + tuple(sorted((k, str(v)) for k, v in opt.items())))

            scores = [self._score_partial(s2, grp, cache) for s2 in succs]
            order = self._rank_successors(scores, exploration, rng)

            # Best-ranked successor whose path prefix is not already taken by another candidate in this batch
            picked_i = None
            for i in order:
                tentative_path = tuple(path_key + [succ_keys[i]])
                if batch_path_tabu is not None and tentative_path in batch_path_tabu:
                    continue
                picked_i = i
                break
            if picked_i is None:
                picked_i = order[0]   # every option is tabu: accept the preferred one anyway

            path_key.append(succ_keys[picked_i])
            if batch_path_tabu is not None:
                batch_path_tabu.add(tuple(path_key))
            cur = succs[picked_i]
            applied.add((finger, phase))

            if phase == "set_servo_count" and int(cur.nodes[finger].get("servo_count", 0)) == 2:
                # No base servo -> grammar_1 is meaningless: assign 0 and skip that phase.
                cur = self.generator.apply_rule(cur, "set_grammar_1", {'finger_idx': finger, 'grammar_1_count': 0})
                applied.add((finger, "set_grammar_1"))
                sched.advance_for(finger, auto_steps=2)
            else:
                sched.advance_for(finger)

            trajectory.append(cur)
            step += 1

        cur.trajectory = [s for s in trajectory if not s.is_complete()]
        return cur

    # ------------------------------------------------------------------ bookkeeping / training
    def _record(self, design: DesignGraph, key: str, group: str, reward: float) -> None:
        """Max-merge a reward into the lookup table and keep one training example per (key, group)."""
        self.lookup_table[(key, group)] = max(self.lookup_table.get((key, group), float('-inf')), float(reward))
        if (key, group) not in self._seen_keys:
            self._seen_keys.add((key, group))
            design._value_key = key
            design.evaluated_group = group
            self.seen_designs.append(design)

    def update_lookup_table(self, design: DesignGraph, reward: float, group: Optional[str]):
        """Record an evaluated hand and credit every partial state on its rollout path with
        max(current, reward) - RoboGrammar's target V*(s) = max over evaluated descendants."""
        grp = group or "default"
        d = canonicalize_fingers(design)
        grp_obj = next((g for g in HandGroup if g.value == grp), None)

        eff_key = getattr(d, "_effective_key", None)
        if not eff_key and grp_obj is not None:
            eff_key = self.compute_effective_key_for_group(d, grp_obj)
        if not eff_key:
            return

        self._record(d, eff_key, grp, reward)
        self.num_evaluated += 1

        if reward <= -999.0:
            return  # a build/evaluation failure says nothing about the ancestors

        for state in getattr(design, "trajectory", None) or []:
            self._record(state, state.partial_state_key(grp), grp, reward)

    def train_value_network(self):
        """Train the value network on all recorded states, per group, toward their lookup-table values."""
        if len(self.seen_designs) < self.batch_size:
            return

        rows_by_group: Dict[str, List[Tuple[DesignGraph, float]]] = {}
        for d in self.seen_designs:
            key = getattr(d, "_value_key", None)
            grp = getattr(d, "evaluated_group", None)
            if key is None or grp is None:
                continue
            v = self.lookup_table.get((key, grp))
            if v is None or v <= -999.0:
                continue
            rows_by_group.setdefault(grp, []).append((d, v))

        trainable = {g: rows for g, rows in rows_by_group.items() if len(rows) >= self.batch_size}
        if not trainable:
            print("⚠️  No groups have enough data for training")
            return

        print(f"🔧 Training GNN on {len(trainable)} groups:")
        for grp, rows in trainable.items():
            print(f"   - {grp}: {len(rows)} states")

        self.value_network.train()
        for group, rows in trainable.items():
            group_opt_iter = max(self.opt_iter // len(trainable), 20)
            print(f"\n   [{group}] Training on {len(rows)} states")

            for step in range(group_opt_iter):
                batch = random.sample(rows, min(self.batch_size, len(rows)))
                loss = 0.0
                for design, target in batch:
                    node_features, edge_indices = design.to_tensor_data(group_name=group)
                    node_features = node_features.to(self.device)
                    edge_indices = edge_indices.to(self.device)
                    predicted = self.value_network(node_features, edge_indices)
                    target_tensor = torch.tensor(float(target), dtype=torch.float32, device=self.device)
                    loss = loss + (predicted - target_tensor) ** 2
                loss = loss / len(batch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
                self.optimizer.step()

                if step % 10 == 0:
                    print(f"      Step {step}/{group_opt_iter}: loss={loss.item():.5f}")

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.99
            print(f"   Learning rate decayed to: {param_group['lr']:.6f}")

    def compute_effective_key_for_group(self, design, group_obj):
        assert self.evaluator is not None, "Set self.evaluator before evaluation"
        d = canonicalize_fingers(design)
        finger_codes, mode, thumb_slot, finger_tips = self.evaluator.effective_codes_for(d, group_obj)
        thumb_str = f"_{thumb_slot}" if thumb_slot is not None else ""
        tip_pairs = ",".join(f"{c}:{t}" for c, t in zip(finger_codes, finger_tips))
        return f"eff_{mode}{thumb_str}_{tip_pairs}"

    # ------------------------------------------------------------------ main loop
    def run(self) -> Tuple[Optional[DesignGraph], float]:
        """Run the Graph Heuristic Search with detailed progress tracking"""
        print(f"\n🚀 STARTING GRAPH HEURISTIC SEARCH")
        print(f"   Iterations: {self.num_iterations}")
        print(f"   Candidates per iteration: {self.num_candidates}")
        print(f"   Total designs to evaluate: {self.num_iterations * self.num_candidates}")
        print(f"   Device: {self.device}")

        search_start_time = time.time()
        initial_epsilon = self.epsilon

        for iteration in range(self.num_iterations):
            progress = iteration / max(1, self.num_iterations - 1)
            self.epsilon = initial_epsilon + (self.epsilon_final - initial_epsilon) * progress

            iter_start_time = time.time()
            print(f"\n🔄 ITERATION {iteration + 1}/{self.num_iterations}")
            print(f"   Exploration ({'tau' if self.selection == 'gumbel' else 'epsilon'}): {self.epsilon:.3f}")
            print(f"   Hands evaluated so far: {self.num_evaluated} (training states: {len(self.seen_designs)})")
            print(f"   Current best score: {self.best_reward:.3f}")

            print(f"   📋 Generating {self.num_candidates} candidate designs...")
            candidates = self.generate_candidate_designs()
            print(f"   ✓ Generated {len(candidates)} candidates")

            print(f"   🔄 Canonicalizing candidates...")
            canonical_candidates = [canonicalize_fingers(candidate) for candidate in candidates]

            print(f"   🎯 Evaluating {len(canonical_candidates)} candidates in parallel...")
            parallel_start_time = time.time()
            iteration_scores = self.evaluator.evaluate_designs_parallel(canonical_candidates)
            parallel_time = time.time() - parallel_start_time
            print(f"   ✓ Parallel evaluation completed in {parallel_time:.1f}s")

            for i, (candidate, score) in enumerate(zip(canonical_candidates, iteration_scores)):
                print(f"   📊 Candidate {i+1}: {candidate.design_string} -> Score: {score:.3f}")

                # The group the evaluator actually used for this design (set by the evaluator).
                group = getattr(candidate, "_effective_group", None) or getattr(candidate, "evaluated_group", None) or "default"
                candidate.evaluated_group = group
                self.update_lookup_table(candidate, score, group)

                if score > self.best_reward:
                    self.best_reward = score
                    self.best_design = candidate
                    print(f"      🏆 NEW BEST DESIGN! Score: {score:.3f}")

            iter_time = time.time() - iter_start_time
            valid_scores = [s for s in iteration_scores if s > -999.0]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

            print(f"\n   📊 Iteration {iteration + 1} Summary:")
            print(f"      Time: {iter_time:.1f}s")
            print(f"      Average score: {avg_score:.3f} ({len(valid_scores)}/{len(iteration_scores)} evaluated)")
            print(f"      Best score this iter: {max(valid_scores) if valid_scores else -1000:.3f}")
            print(f"      Overall best: {self.best_reward:.3f}")

            self.train_value_network()

            if (iteration + 1) % 5 == 0:
                self.save_checkpoint(iteration + 1)
                print(f"      💾 Checkpoint saved for iteration {iteration+1}")

        total_time = time.time() - search_start_time
        print(f"\n🎉 GRAPH HEURISTIC SEARCH COMPLETED!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Hands evaluated: {self.num_evaluated}")
        print(f"   Best score: {self.best_reward:.3f}")
        if self.best_design:
            print(f"   Best design: {self.best_design.design_string}")

        return self.best_design, self.best_reward

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, iteration: int, save_dir: Optional[str] = None):
        """Save current state"""
        save_dir = save_dir or self.checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'iteration': iteration,
            'value_network_state': self.value_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lookup_table': self.lookup_table,
            'best_design': self.best_design,
            'best_reward': self.best_reward,
            'seen_designs': self.seen_designs,
            'num_evaluated': self.num_evaluated,
        }
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{iteration}.pt"))

        if self.best_design:
            with open(os.path.join(save_dir, f"best_design_{iteration}.json"), 'w') as f:
                json.dump({
                    'design_string': self.best_design.design_string,
                    'reward': self.best_reward,
                    'nodes': self.best_design.nodes,
                    'edges': self.best_design.edges
                }, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """Load saved state"""
        # Checkpoints hold pickled DesignGraph objects, so the weights-only loader cannot read them.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.value_network.load_state_dict(checkpoint['value_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.lookup_table = checkpoint['lookup_table']
        self.best_design = checkpoint['best_design']
        self.best_reward = checkpoint['best_reward']
        self.seen_designs = checkpoint['seen_designs']
        self.num_evaluated = checkpoint.get('num_evaluated', 0)
        self._seen_keys = {
            (d._value_key, d.evaluated_group)
            for d in self.seen_designs
            if hasattr(d, "_value_key") and hasattr(d, "evaluated_group")
        }
