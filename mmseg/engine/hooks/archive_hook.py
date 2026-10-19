# Copyright (c) OpenMMLab. All rights reserved.
import subprocess
import sys
from pathlib import Path

from mmengine.hooks import Hook

from mmseg.registry import HOOKS


@HOOKS.register_module()
class ArchiveExperimentHook(Hook):
    """Archive an experiment automatically after a normal training run."""

    def __init__(self,
                 out_dir,
                 config=None,
                 name=None,
                 note=None,
                 val_interval=None,
                 no_weights=False,
                 raise_on_error=False):
        self.repo_root = Path(__file__).resolve().parents[3]
        self.out_dir = out_dir
        self.config = config
        self.name = name
        self.note = note
        self.val_interval = val_interval
        self.no_weights = no_weights
        self.raise_on_error = raise_on_error

    def _resolve(self, path):
        if path is None:
            return None
        path = Path(path)
        if path.is_absolute():
            return path
        return self.repo_root / path

    def after_run(self, runner):
        script = self.repo_root / 'tools/my_tools/archive_experiment.py'
        work_dir = self._resolve(runner.work_dir)
        out_dir = self._resolve(self.out_dir)
        config = self._resolve(self.config)

        cmd = [
            sys.executable,
            str(script),
            '--work-dir',
            str(work_dir),
            '--out-dir',
            str(out_dir),
        ]
        if config is not None:
            cmd.extend(['--config', str(config)])
        if self.name:
            cmd.extend(['--name', self.name])
        if self.note:
            cmd.extend(['--note', self.note])
        if self.val_interval is not None:
            cmd.extend(['--val-interval', str(self.val_interval)])
        if self.no_weights:
            cmd.append('--no-weights')

        runner.logger.info('Archiving experiment after run...')
        result = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            text=True,
            capture_output=True,
            check=False)

        if result.stdout:
            runner.logger.info(result.stdout.strip())
        if result.stderr:
            runner.logger.warning(result.stderr.strip())
        if result.returncode != 0:
            message = f'ArchiveExperimentHook failed with code {result.returncode}'
            if self.raise_on_error:
                raise RuntimeError(message)
            runner.logger.warning(message)
