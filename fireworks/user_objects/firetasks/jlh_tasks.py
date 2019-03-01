#!/usr/bin/env python

# fireworks-internal
from fireworks.core.firework import FWAction, FiretaskBase, Firework, Workflow
from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact
from fireworks.user_objects.firetasks.fileio_tasks import FileTransferTask
from fireworks.user_objects.firetasks.script_task import ScriptTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask

# system
import glob

# from fileio_tasks.py
import os
import shutil
import traceback
# from os.path import expandvars, expanduser, abspath
# import time

# from script_task.py
import shlex
import subprocess
import sys
from fireworks.core.firework import FiretaskBase, FWAction
from six.moves import builtins
if sys.version_info[0] > 2:
    basestring = str


__author__      = 'Johannes Hoermann'
__copyright__   = 'Copyright 2018, IMTEK'
__version__     = '0.1.1'
__maintainer__  = 'Johannes Hoermann'
__email__       = 'johannes.hoermann@imtek.uni-freiburg.de'
__date__        = 'Feb 19, 2019'

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

        stdout = stdout.decode('utf-8') if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode('utf-8') if isinstance(stderr, bytes) else stderr

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
            raise RuntimeError('ScriptTask fizzled! Return code: {}'.format(returncodes))

        return FWAction(stored_data=output)

    def _load_params(self, d):
        if d.get('stdin_file') and d.get('stdin_key'):
            raise ValueError('ScriptTask cannot process both a key and file as the standard in!')

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
            raise ValueError('ScriptTask cannot both FIZZLE and DEFUSE a bad returncode!')

    @classmethod
    def from_str(cls, shell_cmd, parameters=None):
        parameters = parameters if parameters else {}
        parameters['cmd'] = [shell_cmd]
        parameters['use_shell'] = True
        return cls(parameters)


class DummyParentTask(FiretaskBase):
    """
    Reinitiates a workflow chain that has died. Set _launch_dir, _files_out
    to match fizzled firework. Activate _allow_fizzled_parents, _pass_job_info.
    """

    _fw_name = "DummyParentTask"

    def run_task(self, fw_spec):
        spec_to_pass = ['sb_name', 'system_name']
        passed_spec = { key: fw_spec[key] for key in spec_to_pass if key in fw_spec }

        return FWAction(
            update_spec = passed_spec )

class RecoverPackmolTask(FiretaskBase):
    """
    Copies output files of a previously completed or fizzled packmol run
    to a desired destination. Forwards (other or same) files to subsequent
    Fireworks. Set _files_in and _files out to forward files if parent
    successfull. (TO IMPLEMENT: restarts packmol with restart files)

    For packmol, there are several possible outcomes:
      1. packmol finished successfully. The parent Firework has COMPLETED
         and passed on its output file, suffixed .pdb
      2. packmol finished by reaching the maximum amount of iterations,
         but not converged. The parent Firework has COMPLETED and passed
         on two files suffixed .pdb and .pdb_FORCED. The .pdb might contain
         misplaced, overlapping entities, while .pdb_FORCED should at least
         fulfill the constraints, while violating tolerances. In this case,
         forward .pdb_FORCED instead of .pdb.
      3. packmol died (possibly due to insufficient walltime). It did not
         forward any files, but is marked as FIZZLED. two files suffixed
         .pdb and .pdb_FORCED should reside within the fizzled parent's
         launch dir. Pull these files and forward .pdb_FORCED.

    Both case 2. and 3. would allow restart from restart files.
    However, with sufficient walltime it is unlikely to improve packing.
    More likely, the packing is too dense to ever fulfill tolerance
    constraints.

    Args:
        dest: str
           Destination to store packmol output files
        glob_patterns: (list) of str (optional)
           Patterns for glob.glob to identify outputs to store.
           Default: ('*_restart.pack', '*_packmol.pdb',
                     '*_packmol.pdb_FORCED', '*_packmol.inp')
        forward_glob_patterns: {str: str} or {str: list of str} (optional)
           Files to be forwarded to children, either via a detour if parent
           fizzled, or directly if parent successfull and passed via _files_in.
           If value in key: value pair is list of str, preceding glob patterns
           are given priority. Thus, it is possible to forward some .pdb_FORCED
           file if available, and otherwise a .pdb file.
           Default: {"packmol_pdb" : ["*_packmol.pdb_FORCED", "*_packmol.pdb"]}
        recover: bool (optional)
           Pull launchdir from previous Firework if True.
           Otherwise assume output files to have been produced
           in same Firework. Default: False
    """

    _fw_name = "RecoverPackmolTask"
    required_params = ["dest"]

    def run_task(self, fw_spec):
        from glob import glob
        from os.path import join, basename, getmtime
        from os import curdir, getcwd

        recover               = self.get('recover', False)
        glob_patterns         = self.get('glob_patterns', [ '*_restart.pack',
            '*_packmol.pdb', '*_packmol.pdb_FORCED', '*_packmol.inp'] )
        forward_glob_patterns = self.get('forward_glob_patterns',
            { "packmol_pdb" : ["*_packmol.pdb_FORCED", "*_packmol.pdb"] } )

        assert type(recover) is bool, "'recover = {}' is not bool".format(recover)
        assert isinstance(glob_patterns, (str, list)), "'glob_patterns = {}' is neither str not list (of str)".format(glob_patterns)
        assert type(forward_glob_patterns) is dict, "'forward_glob_patterns = {}' is not dict (of str: ( str or (list of str) ) )".format(forward_glob_patterns)

        # check whether a previous firework handed down information
        if recover and ('_job_info' in fw_spec): # pull from intentionally passed job info
            job_info_array = fw_spec['_job_info']
            prev_job_info = job_info_array[-1]
            path_prefix = prev_job_info['launch_dir']
            print('The name of the previous job was: {}'.format(prev_job_info['name']))
            print('The id of the previous job was: {}'.format(prev_job_info['fw_id']))
            print('The location of the previous job was: {}'.format(path_prefix))
        elif recover and ('_fizzled_parents' in fw_spec): # pull from fizzled previous FW
            fizzled_parents_array = fw_spec['_fizzled_parents']
            prev_job_info = fizzled_parents_array[-1] # pull latest (or last) fizzled parent
            path_prefix = prev_job_info['launches'][-1]['launch_dir'] # pull latest launch
            print('The name of fizzled parent Firework was: {}'.format(prev_job_info['name']))
            print('The id of fizzled parent Firework was: {}'.format(prev_job_info['fw_id']))
            print('The location of fizzled parent Firework was: {}'.format(path_prefix))
        elif recover: # no info about previous (fizzled or other) jobs given
            path_prefix = getcwd()
            print('No information about previous (fizzled or other) jobs available.')
            print('Checking for files in specs._files_in...')
            if '_files_in' in fw_spec:
                print('{} are expected from a (successfull) parent'.format(
                        fw_spec['_files_in']))
            else:
                print('No _files_in, nothing to be done. Exiting successfully.')
                return FWAction()
        else:
            path_prefix = getcwd() # assume all output files in current directory
            print('Take output files from cwd.')

        files_to_copy = []
        if type(glob_patterns) is not list:
            glob_patterns = [ glob_patterns ]
        for glob_pattern in glob_patterns:
            files_to_copy.extend(
                glob(
                    join( path_prefix, glob_pattern )
                )
            )

        files_to_forward = {}
        for key, glob_pattern_list in forward_glob_patterns.items():
            if glob_pattern_list is str:
                glob_pattern_list = [ glob_pattern_list ]
            assert isinstance(glob_pattern_list, list), "Value of item '{}: {}' in forward_glob_patterns neither str nor list (of str).".format(key, glob_pattern_list)
            for glob_pattern in glob_pattern_list:
                assert isinstance(glob_pattern, str), "Item '{}' in value of item '{}: {}' in forward_glob_patterns not str.".format(glob_pattern, key, glob_pattern_list)
                file_list = sorted( glob( join( path_prefix, glob_pattern ) ), key = getmtime )
                if len( file_list ) < 1:
                    print("No file to forward found for {:s} --> {:s} globbing.".format(key, glob_pattern))
                    continue

                if len( file_list ) > 1:
                    print("ATTENTION: {:s} --> {:s} globbing yielded more than one file ( {} ).".format(key, glob_pattern, file_list ) )
                    print("Only using last entry (newest, sorted by modification time).")
                files_to_forward[key] = file_list[-1]
                break

            if key not in files_to_forward: # no file found for any of the glob patterns in list
                raise ValueException("No file to forward found for any of {:s} --> {} globbing.".format(key, glob_pattern_list))

        print('Files to be stored:    {}'.format(files_to_copy))
        print('Files to be forwarded: {}'.format(files_to_forward))

        # prepend system_name to fw name if available
        store_packmol_files_fw_name   = "store_packmol_files"
        forward_packmol_files_fw_name = "forward_packmol_files"

        # spec to be inherited by dynamically created FW
        spec_to_pass_on = [ '_queueadapter', '_category',
            'system_name', 'sb_name', 'geninfo']
        passed_spec = {}
        for spec in spec_to_pass_on:
            if spec in fw_spec:
                passed_spec[spec] = fw_spec[spec]
                print("spec '{}: {}' passed on to children".format(spec, passed_spec[spec]))

        if 'system_name' in fw_spec:
            store_packmol_files_fw_name = '_'.join(
                [fw_spec['system_name'],store_packmol_files_fw_name])
            forward_packmol_files_fw_name = '_'.join(
                [fw_spec['system_name'],forward_packmol_files_fw_name])

        additional_fw = Firework(
            FileTransferTask(
                {
                    'files':         files_to_copy,
                    'dest':          self.get('dest'),
                    'mode':          'copy',
                    'ignore_errors': True
                } ),
            spec = dict(passed_spec, _dupefinder=DupeFinderExact() ),
            name = store_packmol_files_fw_name )

        files_to_forward_basename = { key: basename(file) for key, file in files_to_forward.items() }

        detour_fw = Firework(
            FileTransferTask(
                {
                    'files':         list(files_to_forward.values()),
                    'dest':          curdir,
                    'mode':          'copy',
                    'ignore_errors': False
                } ),
            spec = dict( passed_spec,
                _dupefinder =   DupeFinderExact(),
                _files_out =    files_to_forward_basename # is dict
            ),
            name = forward_packmol_files_fw_name)

        return FWAction(
            additions   = additional_fw,
            detours     = detour_fw )

class RecoverLammpsTask(FiretaskBase):
    """
    Reinitiates a LAMMPS run that has died (probably from exceeded walltime).
    Activate _allow_fizzled_parents and append to LAMMPS run Firework.
    Will identify most recent restart file and initiate another LAMMPS run.

    Runs any python function! Extremely powerful, which allows you to
    essentially run any accessible method on the system. The optional inputs
    and outputs lists may contain spec keys to add to args list and to make
    the function output available in the curent and in children fireworks.

    Required parameters:

    Optional parameters:
        - recover (bool):
          Pull launchdir from previous Firework if True.
          Otherwise assume output files to have been produced
          in same Firework. Default: True
        # - restart_fw (dict or [dict]):
        #   Single firework or list of Fireworks to append
        #   (i.e. as LAMMPS restart run to continue from restart file).
        #   If list, all fireworks will be independent of each other (no links).
        #   Task will not append anything if None. Default: None
        - restart_fw (dict):
          Single firework to append only if restart file present (parent failed)
          (i.e. as LAMMPS restart run to continue from restart file).
          Task will not append anything if None. Default: None
        - detour_fw (dict):
          Single firework to append always, independent on parent's success.
          (i.e.  LAMMPS post-processing).
          Task will not append anything if None. Default: None
        # Not implemented:
        # - addition_wf (dict):
        #   Workflow to append as addition. Default: None
        - max_restart (int):
          Maximum number of repeated restarts (in case of 'restart' == True)
          Default: 5
        - restart_file_glob_patterns (str or [str])
          Patterns for glob.glob to identify restart files.
          Attention: Be careful not to match any restart file that has been
          used as an input initially. Default: ['*.restart[0-9]']
        - other_glob_patterns (str or [str])
          Patterns for glob.glob to identify other files to be forwarded.
          Default: None
        - default_restart_file (str):
          Name of restart file to be read by subsequent LAMMPS restart run.
          Most recent restart file found in fizzled parent will be copied
          to new launchdir under this name.
          Default: 'default.mpiio.restart'
        - fizzle_on_no_restart_file (bool):
          Default: True
        - repeated_recover_fw_name (str):
          Name for repeated recovery fireworks.
          Default: 'Repeated LAMMPS recovery'
        - ignore_errors (bool):
          Ignore errors when copying files. Default: True
        # Not implemented, always copied:
        # - mode ('copy' or 'forward'):
        #   Either copy desired files to current directory, or just forward
        #   them via setting '_files_prev' accordingly.
        #   Default: 'forward'
    """
    _fw_name = "RecoverLammpsTask"
    required_params = []
    optional_params = [
        "recover",
        "restart",
        "restart_fw",
        #"addition_wf",
        "max_restarts",
        "restart_file_glob_patterns"
        "other_glob_patterns",
        "default_restart_file",
        "fizzle_on_no_restart_file",
        "repeated_recover_fw_name",
        "mode" ]

    def run_task(self, fw_spec):
        from os import path, curdir, getcwd

        recover                    = self.get('recover', True)
        restart_fw_dict            = self.get('restart_fw', None)
        detour_fw_dict             = self.get('detour_fw', None)
        #addition_wf_dict           = self.get('addition_wf', None)
        max_restarts               = self.get('max_restarts', 5)
        repeated_recover_fw_name   = self.get('repeated_recover_fw_name',
            'Repeated LAMMPS recovery')
        restart_file_glob_patterns = self.get('restart_file_glob_patterns', ['*.restart[0-9]'] )
        default_restart_file       = self.get('default_restart_file', 'default.mpiio.restart')
        other_glob_patterns        = self.get('other_glob_patterns', None )
        fizzle_on_no_restart_file  = self.get('fizzle_on_no_restart_file', True)
        ignore_errors              = self.get('ignore_errors', True)
        # mode                       = self.get('mode', 'forward')

        # check whether a previous firework handed down information
        prev_job_info = None
        if recover and ('_job_info' in fw_spec): # pull from intentionally passed job info
            job_info_array = fw_spec['_job_info']
            prev_job_info = job_info_array[-1]
            path_prefix = prev_job_info['launch_dir']
            print('The name of the previous job was: {}'.format(prev_job_info['name']))
            print('The id of the previous job was: {}'.format(prev_job_info['fw_id']))
            print('The location of the previous job was: {}'.format(path_prefix))
        elif recover and ('_fizzled_parents' in fw_spec): # pull from fizzled previous FW
            fizzled_parents_array = fw_spec['_fizzled_parents']
            prev_job_info = fizzled_parents_array[-1] # pull latest (or last) fizzled parent
            path_prefix = prev_job_info['launches'][-1]['launch_dir'] # pull latest launch
            print('The name of fizzled parent Firework was: {}'.format(prev_job_info['name']))
            print('The id of fizzled parent Firework was: {}'.format(prev_job_info['fw_id']))
            print('The location of fizzled parent Firework was: {}'.format(path_prefix))
        elif recover: # no info about previous (fizzled or other) jobs given
            print('No information about previous (fizzled or other) jobs available.')
            # completed successfully?
            # return FWAction()
        else:
            path_prefix = getcwd() # assume all output files in current directory
            print('Work on own launch dir.')
        # only recover, if prev_job info not None

        # find other files to forward:
        file_list = []
        if type(other_glob_patterns) is not list:
            other_glob_patterns = [ other_glob_patterns ]
        for other_glob_pattern in other_glob_patterns:
            if type(restart_file_glob_pattern) is str: # avoid non string objs
                print("Processing glob pattern {}".format(other_glob_pattern))
                file_list.extend(
                    glob.glob( path.join( path_prefix, other_glob_pattern )  )
                )

        # copy other files if necessary
        if len(file_list) > 0:
            for f in file_list:
                print("File {} will be forwarded.".format(f))
                try:
                    dest = os.getcwd()
                    shutil.copy(f,dest)
                except:
                    traceback.print_exc()
                    if not ignore_errors:
                        raise ValueError("There was an error "
                            "copying '{}' to '{}'.".format(f, dest))
                    else:
                        print("There was an error "
                            "copying '{}' to '{}', ignored.".format(f, dest))

        # find restart files:
        restart_file_list = []
        # avoid iterating through each character of string
        if type(restart_file_glob_patterns) is not list:
            restart_file_glob_patterns = [ restart_file_glob_patterns ]
        for restart_file_glob_pattern in restart_file_glob_patterns:
            if type(restart_file_glob_pattern) is str: # avoid non string objs
                restart_file_list.extend(
                    glob.glob( path.join( path_prefix, restart_file_glob_pattern ) )
                )

        # determine most recent of all restart files:
        if len(restart_file_list) > 1:
            sorted_restart_file_list = sorted(
                restart_file_list, key = path.getmtime) # sort by modification time
            print("Several restart files: {} (most recent last)".format(sorted_restart_file_list))
            current_restart_file = sorted_restart_file_list[-1]
        elif len(restart_file_list) == 1:
            print("One restart file: {}".format(restart_file_list[0]))
            current_restart_file = restart_file_list[-1]
        else:
            print("No restart file!")
            current_restart_file = None
            if fizzle_on_no_restart_file:
                 raise ValueError("No restart file in {:s}".format(path_prefix))

        detour_wf = None # everything goes into detour workflow for now
        addition_wf = None

        detour_fws = []
        detour_fws_links = {}

        # if detour_fw given, append in any case:
        consecutive_fw_id = -1 # quite an ugly necessity
        if detour_fw_dict:
            # append firework (defined as dict):
            detour_fw = Firework.from_dict(detour_fw_dict)
            detour_fw.fw_id = consecutive_fw_id
            consecutive_fw_id = consecutive_fw_id - 1
            # manually set _files_prev:
            # detour_fw.spec["_files_prev"] = update_spec["_files_prev"]

            # check whether this Firework's _files_out can serve subsequent
            # Firework's _files_in:
            files_prev = {}
            if ("_files_out" in fw_spec) and (type(fw_spec["_files_out"]) is dict) \
                and ("_files_in" in detour_fw.spec) \
                and (type(detour_fw.spec["_files_in"]) is dict):

                print("Current recovery '_files_out': {}".format(fw_spec["_files_out"]))
                print("Subsequent detour 'files_in': {}".format(detour_fw.spec["_files_in"]))

                # find overlap between current fw's outfiles and detour fw's
                # infiles:
                files_io_overlap = fw_spec["_files_out"].keys() \
                    & detour_fw.spec["_files_in"].keys()
                print("Current recovery '_files_out' and subsequent detour "
                    "'files_in' overlap: {}".format(files_io_overlap))

                for k in files_io_overlap:
                    files = glob.glob(os.path.join(curdir,
                        fw_spec["_files_out"][k]))
                    if files:
                        files_prev[k] = os.path.abspath(sorted(files)[-1])
                        print("This Firework provides {}: {}".format(
                            k, fw_spec["_files_out"][k] ), " for subsequent "
                            "detour Firework.")

            detour_fw.spec["_files_prev"] = files_prev
            print("Create detour Firework {} with id {} and specs {}".format(
                    detour_fw.name, detour_fw.fw_id, detour_fw.spec) )
            detour_fws.append(detour_fw)

        # append restart fireworks if restart file exists
        if current_restart_file is not None and restart_fw_dict:
            current_restart_file_basename = path.basename(current_restart_file)
            print("File {} will be forwarded.".format(
                current_restart_file_basename))

            try:
                shutil.copy(current_restart_file, default_restart_file)
            except:
                traceback.print_exc()
                raise ValueError(
                    "There was an error copying from {} "
                    "to {}".format(current_restart_file, default_restart_file) )

            # try to derive number of restart from fizzled parent
            if prev_job_info and 'spec' in prev_job_info \
                and 'restart_count' in prev_job_info['spec']:
                restart_count = \
                    int(prev_job_info['spec']['restart_count']) + 1
            else:
                restart_count = 0

            if restart_count < max_restarts:
                print("This is restart #{:d} of at most {:d} restarts.".format(
                    restart_count+1, max_restarts ) )

                restart_fw = Firework.from_dict(restart_fw_dict)
                restart_fw.fw_id = consecutive_fw_id
                consecutive_fw_id = consecutive_fw_id - 1
                # append restart firework (defined as dict):
                restart_fw.spec["restart_count"] = restart_count


                # copied from fireworks.core.rocket.decorate_fwaction
                # for some reason, _files_prev is not updated automatically when
                # returning the raw FWAction object.
                # Possibly, update_spec or mod_spec is evaluated before
                # additions and detours are appended?
                files_prev = {}
                if "_files_out" in fw_spec and type(fw_spec["_files_out"]) is dict \
                    and "_files_in" in restart_fw.spec \
                    and type(restart_fw.spec["_files_in"]) is dict:
                    print("Current recovery '_files_out': {}".format(fw_spec["_files_out"]))
                    print("Subsequent restart 'files_in': {}".format(restart_fw.spec["_files_in"]))
    
                    # find overlap between current fw's outfiles and restart fw's
                    # infiles:
                    files_io_overlap = fw_spec["_files_out"].keys() \
                        & restart_fw.spec["_files_in"].keys()
                    print("Current recovery '_files_out' and subsequent restart "
                        "'files_in' overlap: {}".format(files_io_overlap))
    
                    for k in files_io_overlap:
                        files = glob.glob(os.path.join(curdir,
                            fw_spec["_files_out"][k]))
                        if files:
                            files_prev[k] = os.path.abspath(sorted(files)[-1])
                            print("This Firework provides {}: {}".format(
                                k, fw_spec["_files_out"][k] ), " for subsequent "
                                "restart Firework." )
    
                # manually set _files_prev:
                restart_fw.spec["_files_prev"] = files_prev

                print("Create restart Firework {} with id {} and specs {}".format(
                    restart_fw.name, restart_fw.fw_id, restart_fw.spec) )

                # repeatedly append copy of this recover task:
                recover_ft = self

                # repeated recovery firework inherits the following specs:
                spec_to_pass = [ '_allow_fizzled_parents', '_category',
                    '_files_in', '_files_out', '_tasks' ]
                recover_fw_spec = { key: fw_spec[key] for key in spec_to_pass if key in fw_spec }

                recover_fw = Firework(
                    recover_ft,
                    spec = recover_fw_spec, # inherit thi Firework's spec
                    name = repeated_recover_fw_name,
                    parents = [ restart_fw ],
                    fw_id = consecutive_fw_id  )
                consecutive_fw_id = consecutive_fw_id - 1
                print("Create repeated recover Firework {} with id {} and specs {}".format(
                    recover_fw.name, recover_fw.fw_id, recover_fw.spec) )

                detour_fws.append(restart_fw)

                # make repeated recovery fireworks dependent on all other
                # fireworks (i.e. detour and restart) in list:
                for fw in detour_fws:
                    detour_fws_links[fw] = recover_fw

                detour_fws.append(recover_fw)

                print("Links in detour: {}".format(detour_fws_links) )
            else:
                print("Maximum number of {:d} restarts reached. ".format(
                    restart_count+1, max_restarts ),  "No further restart.")
        else:
             raise ValueError("No restart file (should never reach this point)")

        # detour_fws can be empty:
        if len(detour_fws) > 0:
            detour_wf = Workflow( detour_fws, detour_fws_links )
        return FWAction(
            additions   = addition_wf, # dummy for now
            detours     = detour_wf)  # will forward restart file if wished for

class MakeSegIdSegPdbDictTask(FiretaskBase):
    """
    Simply globs all files matching a certain pattern and assigns a 4-letter ID to each of them.

    Args:
        glob_pattern (str)
    """

    _fw_name = "MakeSegIDSegPDBDictTask"
    required_params = ["glob_pattern"]

    def run_task(self, fw_spec):
        from glob import glob

        def fourLetterIDfromInt(n):
            return chr( (n // 26**3) % 26 + ord('A') ) \
                 + chr( (n // 26**2) % 26 + ord('A') ) \
                 + chr( (n // 26**1) % 26 + ord('A') ) \
                 + chr( (n // 26**0) % 26 + ord('A') )

        def fourLetterID(max=456976): # 26**4
            """generator for consectutive 4 letter ids"""
            for n in range(0,max):
                yield fourLetterIDfromInt(n)

        glob_pattern = self.get('glob_pattern', '*_[0-9][0-9][0-9].pdb')
        alpha_id = fourLetterID()

        seg_id_seg_pdb_dict = {
            next(alpha_id): pdb for pdb in sorted( glob(glob_pattern) ) }

        print("Found these segments: {}".format(seg_id_seg_pdb_dict))

        return FWAction(
            stored_data = { 'seg_id_seg_pdb_dict' : seg_id_seg_pdb_dict},
            mod_spec = [ { '_set' : { 'context->segments' : seg_id_seg_pdb_dict } } ] )

class RecoverFilesFromFizzledParentTask(FiretaskBase):
    """
    Activate _allow_fizzled_parents and append to (possibly fizzled ) Firework.
    Will copy desired files from parent directory if existant.

    Args:
        glob_patterns: (list of) str
          Patterns for glob.glob to identify files.
        fizzle_on_no_file: bool (optional)
          Default: False
        ignore_errors: bool (optional)
          Ignore errors when copying files. Default: True
        shell_interpret: bool (optional)
          Expand environment variables and other placeholders in filenames.
          Default: True
    """

    _fw_name = "RecoverFilesFromFizzledParentTask"
    required_params = [ "glob_patterns" ]
    optional_params = ["fizzle_on_no_file", "ignore_errors", "shell_interpret"]

    def run_task(self, fw_spec):
        from os import path, getcwd
        from os.path import expandvars, expanduser, abspath
        import shutil

        glob_patterns      = self.get('glob_patterns')
        fizzle_on_no_file  = self.get('fizzle_on_no_file', False)
        ignore_errors      = self.get('ignore_errors', True)
        shell_interpret    = self.get('shell_interpret', True)

        # check whether a previous firework handed down information
        prev_job_info = None
        if '_job_info' in fw_spec: # pull from intentionally passed job info
            job_info_array = fw_spec['_job_info']
            prev_job_info = job_info_array[-1]
            path_prefix = prev_job_info['launch_dir']
            print('The name of the previous job was: {}'.format(prev_job_info['name']))
            print('The id of the previous job was: {}'.format(prev_job_info['fw_id']))
            print('The location of the previous job was: {}'.format(path_prefix))
        elif '_fizzled_parents' in fw_spec: # pull from fizzled previous FW
            fizzled_parents_array = fw_spec['_fizzled_parents']
            prev_job_info = fizzled_parents_array[-1] # pull latest (or last) fizzled parent
            path_prefix = prev_job_info['launches'][-1]['launch_dir'] # pull latest launch
            print('The name of fizzled parent Firework was: {}'.format(prev_job_info['name']))
            print('The id of fizzled parent Firework was: {}'.format(prev_job_info['fw_id']))
            print('The location of fizzled parent Firework was: {}'.format(path_prefix))
        else: # no info about previous (fizzled or other) jobs given
            print('No information about previous (fizzled or other) jobs available.')
            print('Nothing to be done. Exiting successfully.')
            return FWAction()

        file_list = []
        # avoid iterating through each character of string
        if type(glob_patterns) is not list:
            glob_patterns = [ glob_patterns ]

        for glob_pattern in glob_patterns:
            print("Processing glob pattern {}".format(glob_pattern))
            file_list.extend(
                glob.glob( path.join( path_prefix, glob_pattern ) )
            )

        if len(file_list) < 1 and fizzle_on_no_file:
            raise ValueError("No file found in parent's launch directory!")

        # from fileio_tasks.py
        for f in file_list:
            try:
                src = abspath(expanduser(expandvars(f))) if shell_interpret else f
                dest = getcwd()
                shutil.copy(src,dest)
            except:
                traceback.print_exc()
                if not ignore_errors:
                    raise ValueError("There was an error "
                        "copying '{}' to '{}'.".format(src, dest))

        return FWAction()
