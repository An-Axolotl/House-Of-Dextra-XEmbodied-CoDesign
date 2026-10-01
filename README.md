# House of Dextra: Cross-Embodied Co-Design for Dexterous Hands

![House of Dextra](header.gif)

House of Dextra is the unified repository for the dexterous hand co-design stack used in *House of Dextra: Cross-Embodied Co-Design for Dexterous Hands*.

At a high level, the project combines:
- morphology generation for candidate hand designs,
- simulation-based policy evaluation and search,
- and real-world control/deployment tooling.

**Published at ICLR 2026.** [[arXiv](https://arxiv.org/abs/2512.03743)] [[OpenReview](https://openreview.net/pdf?id=k8ovuXEQQu)] [[Hugging Face](https://huggingface.co/papers/2512.03743)]

For full method details, experiment setup, and results, check out [our paper](https://arxiv.org/abs/2512.03743).

For videos and hardware build guide, please see [our website](https://an-axolotl.github.io/HouseofDextra/index.html).

## Repository Map

- [`Main/`](Main/): graph-heuristic search loop and experiment orchestration.
- [`IsaacLab/`](IsaacLab/): simulation environments, policies, and task integration.
- [`Generation/`](Generation/): hand asset generation and conversion pipeline.
- [`RealControl/`](RealControl/): real hardware control and deployment utilities.

## Read This Next

The top-level README is intentionally high-level. Use the setup guide and directory READMEs for workflow-specific instructions:
- [`SETUP.md`](SETUP.md)
- [`Main/README.md`](Main/README.md)
- [`IsaacLab/README.md`](IsaacLab/README.md)
- [`Generation/README.md`](Generation/README.md)
- [`RealControl/README.md`](RealControl/README.md)

Setup, prerequisites, and Isaac Sim installation steps are documented in [`SETUP.md`](SETUP.md).

## Citation

If you use this in your research, please cite:

```bibtex
@inproceedings{fay2026houseofdextra,
  title={House of Dextra: Cross-Embodied Co-Design for Dexterous Hands},
  author={Fay, Kehlani and Djapri, Darin and Zorin, Anya and Clinton, James
          and El Lahib, Ali and Su, Hao and Tolley, Michael T. and Yi, Sha
          and Wang, Xiaolong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://arxiv.org/abs/2512.03743}
}
```
