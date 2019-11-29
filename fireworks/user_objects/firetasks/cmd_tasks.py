#!/usr/bin/env python

# fireworks-internal
from fireworks.core.firework import FWAction, FiretaskBase

import os
# from script_task.py
import subprocess
import sys
if sys.version_info[0] > 2:
    basestring = str


__author__      = 'Johannes Hoermann'
__copyright__   = 'Copyright 2018, IMTEK'
__version__     = '0.1.1'
__maintainer__  = 'Johannes Hoermann'
__email__       = 'johannes.hoermann@imtek.uni-freiburg.de'
__date__        = 'Nov 29, 2019'

class CmdTask(FiretaskBase):
    """ Enhanced script task, runs (possibly environment dependent)  command.
    Makes use of '_fw_env' worker specific definitions and allow for certain
    abstraction when running commands, i.e.

    - _fw_name: CmdTask
      cmd: lmp
      opt:
      - -in lmp_production.input
      - -v surfactant_name SDS
      - -v has_indenter 1
      - -v constant_indenter_velocity -0.0001
      - -v productionSteps 375000
      [...]
      - -v store_forces 1
      stderr_file:   std.err
      stdout_file:   std.out
      fizzle_bad_rc: true
      use_shell:     true

    will actually look for a definition 'lmp' within a worker's '_fw_env' and
    if available execute in place of the  specififed command, simply appending
    the list of options given in 'opt' separated by sapces. This allows machine-
    specific cod to be placed within the worker files, i.e. for a machine called
    NEMO within nemo_queue_worker.yaml

    name: nemo_queue_worker
    category: [ 'nemo_queue' ]
    query: '{}'
    env:
      lmp: module purge;
           module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;
           module use ${HOME}/modulefiles;
           module load lammps/16Mar18-gnu-7.3-openmpi-3.1-colvars-09Feb19;
           mpirun ${MPIRUN_OPTIONS} lmp

    and for a machine called JUWELS within

    name:     juwels_queue_worker
    category: [ 'juwels_queue' ]
    query:    '{}'
    env:
      lmp:  module purge;
            jutil env activate -p chfr13;
            module use ${PROJECT}/hoermann/modules/modulefiles;
            module load Intel/2019.0.117-GCC-7.3.0 IntelMPI/2018.4.274;
            module load jlh/lammps/16Mar18-intel-2019-gcc-7.3-impi-2018-colvars-18Feb19;
            srun lmp

    A third machine's worker file might look like this:

    name: bwcloud_std_fworker
    category: [ 'bwcloud_std', 'bwcloud_noqueue' ]
    query: '{}'
    env:
      lmp:  module purge;
            module load LAMMPS;
            mpirun -n 4 lmp
      vmd:  module purge;
            module load VMD;
            vmd
      pizza.py: module load MDTools/jlh-25Jan19-python-2.7;
                pizza.py

    This allows for machine-independent workflow design.
    """
    required_params = ['cmd']
    _fw_name = 'CmdTask'

    def run_task(self, fw_spec):
        if self.get('use_global_spec'):
            self._load_params(fw_spec)
        else:
            self._load_params(self)

        # get the standard in and run task internally
        if self.stdin_file:
            with open(self.stdin_file) as stdin_f:
                return self._run_task_internal(fw_spec, stdin_f)
        stdin = subprocess.PIPE if self.stdin_key else None
        return self._run_task_internal(fw_spec, stdin)

    def _run_task_internal(self, fw_spec, stdin):
        # run LAMMPS
        stdout = subprocess.PIPE if self.store_stdout or self.stdout_file else None
        stderr = subprocess.PIPE if self.store_stderr or self.stderr_file else None
        returncodes = []

        if "_fw_env" in fw_spec and self.cmd in fw_spec["_fw_env"]:
            # check whether there is any desired command and whether there
            # exists a machine-specific "alias"
            cmd = fw_spec["_fw_env"][self.cmd]
        else:
            cmd = self.cmd

        if self.opt: # append command line options if available
            cmd = ' '.join((cmd,*self.opt))

        p = subprocess.Popen(
            cmd, executable=self.shell_exe, stdin=stdin,
            stdout=stdout, stderr=stderr,
            shell=self.use_shell)

            # communicate in the standard in and get back the standard out and returncode
        if self.stdin_key:
            (stdout, stderr) = p.communicate(fw_spec[self.stdin_key])
        else:
            (stdout, stderr) = p.communicate()
        returncodes.append(p.returncode)

        # write out the output, error files if specified

        stdout = stdout.decode('utf-8', errors="ignore") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode('utf-8', errors="ignore") if isinstance(stderr, bytes) else stderr

        if self.stdout_file:
            with open(self.stdout_file, 'a+') as f:
                f.write(stdout)

        if self.stderr_file:
            with open(self.stderr_file, 'a+') as f:
                f.write(stderr)

        # write the output keys
        output = {}

        if self.store_stdout:
            output['stdout'] = stdout

        if self.store_stderr:
            output['stderr'] = stderr

        output['returncode'] = returncodes[-1]
        output['all_returncodes'] = returncodes

        if self.defuse_bad_rc and sum(returncodes) != 0:
            return FWAction(stored_data=output, defuse_children=True)

        elif self.fizzle_bad_rc and sum(returncodes) != 0:
            raise RuntimeError('CmdTask fizzled! Return code: {}'.format(returncodes))

        return FWAction(stored_data=output)

    def _load_params(self, d):
        if d.get('stdin_file') and d.get('stdin_key'):
            raise ValueError('CmdTask cannot process both a key and file as the standard in!')

        self.use_shell = d.get('use_shell', True)

        self.cmd = d.get('cmd')

        # command line options
        opt = d.get('opt',None)
        if isinstance(opt, basestring):
            opt = [opt]
        self.opt = opt

        self.stdin_file = d.get('stdin_file')
        self.stdin_key = d.get('stdin_key')
        self.stdout_file = d.get('stdout_file')
        self.stderr_file = d.get('stderr_file')
        self.store_stdout = d.get('store_stdout')
        self.store_stderr = d.get('store_stderr')
        self.shell_exe = d.get('shell_exe', '/bin/bash') # bash as default
        self.defuse_bad_rc = d.get('defuse_bad_rc')
        self.fizzle_bad_rc = d.get('fizzle_bad_rc', not self.defuse_bad_rc)


        if self.defuse_bad_rc and self.fizzle_bad_rc:
            raise ValueError('CmdTask cannot both FIZZLE and DEFUSE a bad returncode!')

    @classmethod
    def from_str(cls, shell_cmd, parameters=None):
        parameters = parameters if parameters else {}
        parameters['cmd'] = [shell_cmd]
        parameters['use_shell'] = True
        return cls(parameters)
