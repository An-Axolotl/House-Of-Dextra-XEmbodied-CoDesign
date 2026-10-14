Changelog
---------

0.2.0 (2026-10-14)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Reorientation env: per-design episodes are counted once (the initial reset no longer counts, and
  the duplicate roll-up in ``play.py`` is gone), so average rewards are no longer divided by three.
* Reorientation env: observation scaling clamps the joint range (ghost joints), non-finite rewards are
  zeroed per env, and ``unstable_episode_fraction`` is reported per design.
* Reorientation env: ``CODESIGN_FIXED_DR_LEVEL`` and ``CODESIGN_DISABLE_ADR_ADAPTATION`` are honored
  (``LeapHandADR.set_level``).
* ``play.py``: ``evaluation_truncated`` / ``steps_completed`` in the final payload; non-finite log
  values are dropped before the JSON dump; checkpoint/observation size mismatches raise a clear error.

Added
^^^^^

* Optional fingertip conditioning of the morphology vector (``CODESIGN_MORPH_FINGERTIPS=1``), parsed
  from the ``_t<letters>`` token in generated hand names.

0.1.0 (2025-07-31)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab