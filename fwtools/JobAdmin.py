# computational
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import scipy as scp
#import scipy.interpolate
import scipy.constants as C
#import ase.data

# formats & visualization
# import parmed as pmd
# from parmed import gromacs
# visualization
#from ase.visualize import view
# import nglview as nv

# PrettyPrint
# from pprint import pprint

# for dictionary comparison:
# from deepdiff import DeepDiff
# https://stackoverflow.com/questions/5903720/recursive-diff-of-two-python-dictionaries-keys-and-values

# system basics
import logging
import os
import sys
import shutil
import glob
import subprocess
import datetime
import importlib

# fireworks
import pymongo
from fireworks import Firework, LaunchPad, Workflow, FWorker, FireTaskBase
from fireworks.core.rocket_launcher import rapidfire, launch_rocket
from fireworks.queue.queue_adapter import QueueAdapterBase
from fireworks.queue.queue_launcher import launch_rocket_to_queue

from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact
from fireworks.user_objects.firetasks.fileio_tasks import FileTransferTask, FileWriteTask
from fireworks.user_objects.firetasks.script_task import ScriptTask, PyTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask, GetFilesTask, DeleteFilesTask

# own tasks
from fireworks.user_objects.firetasks.jlh_tasks import MakeSegIdSegPdbDictTask
from fireworks.user_objects.firetasks.jlh_tasks import RecoverLammpsTask
from fireworks.user_objects.firetasks.jlh_tasks import RecoverPackmolTask
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter

# for testing
from fireworks.features.stats import FWStats
from fireworks.utilities.filepad import FilePad, LAUNCHPAD_LOC

# hard-coded system-sepcific
# sds length, from head sulfur to tail carbon
l_SDS = 14.0138 # Angstrom
# measured on NEMO with
# module load vmd/1.9.3-text
# vmd
#   source distance.tcl
#   distance "name S" "name C12" 1 dist.txt hist.txt
#   exit
# cat dist.txt

# atom  1:   S, in head group
# atom 39: C12, in tail
head_atom_number_SDS = 1
tail_atom_number_SDS = 39

# ctab length, from head nitrogen to tail carbon
l_CTAB = 19.934 # Angstrom

# atom 17: N1, in head group
# atom  1: C1, in tail
head_atom_number_CTAB = 17
tail_atom_number_CTAB = 1


# Tolerance in packmol
tolerance = 2 # Angstrom

# logger = logging.getLogger(__name__)

class JobAdmin:
    """Toolbox for Fireworks on NEMO and JUWELS"""
    def __init__( self,
        mongodb_host = 'localhost',
        mongodb_port = 27017,
        mongodb_name = 'fireworks-testuser',
        mongodb_user = 'fireworks',
        mongodb_pwd  = 'fireworks',
        lmp_cmd = ' '.join((
            'module use /gpfs/homea/hka18/hka184/modules/modulefiles;',
            'module load jlh/lammps/16Mar18-intel;',
            'srun lmp -in {inputFile:s}' )),
        queue           = 'JUWELS',
        template_prefix = '/gpfs/homea/hka18/hka184/jobs/lmplab/sds/201808/N_SDS_on_AU_111_template',
        output_prefix   = '/gpfs/homea/hka18/hka184/jobs/lmplab/sds/201808/sys',
        sim_df = None ):
        '''Establishes connection to Fireworks MongoDB.'''

        self.lpad = LaunchPad(
            host=mongodb_host,
            port=mongodb_port,
            name=mongodb_name,
            username=mongodb_user,
            password=mongodb_pwd)

        self.fp   = FilePad(
            host=mongodb_host,
            port=mongodb_port,
            database=mongodb_name,
            username=mongodb_user,
            password=mongodb_pwd)

        self.template_prefix    = template_prefix
        self.output_prefix      = output_prefix

        self.lmp_cmd            = lmp_cmd
        self._template_lmp_cmd  = lmp_cmd
        self._geninfo = None

        if queue == 'JUWELS':
            self.use_juwels_queue()
            self.use_juwels_templates()
        elif queue == 'NEMO':
            self.use_nemo_queue()
            self.use_nemo_templates()
        elif queue == 'BWCLOUD':
            self.use_bwcloud_queue()
            self.use_bwcloud_templates()
        else:
            raise ValueError("Queue '{:s}' does not exist!".format(queue))

        self._set_machine_independent_templates()

        if sim_df is not None:
            self._sim_df = sim_df

    def bryan_test(self):
        print("New test function added on 29th Jan 2019")

    def use_bwcloud_templates(self):
        # TODO: find way to wrap environment elegantly elsewhere
        self._template_sb_replicate_cmd = ' '.join((
            'module load GROMACS GROMACS-Top MDTools;',
            'replicate.sh {multiples[0]:d} {multiples[1]:d}',
            '{multiples[2]:d} {plane:03d}'))

        self._template_packmol_cmd = ' '.join((
            'module load MDTools;',
            'packmol < {infile:s}' ))

        self._template_packmol2gmx_cmd  = ' '.join((
            'module load MDTools;',
            'pdb_packmol2gmx.sh {infile:s}' ))

        self._template_gmx2pdb_cmd = ' '.join((
            'module load GROMACS GROMACS-Top VMD;',
            'bash {infile:s}' ))

        # psfgen
        self._template_psfgen_cmd  = ' '.join((
            'module load VMD;',
            'vmd -e {infile:s}' ))

        # charmm2lammps.pl
        self._template_ch2lmp_cmd  = ' '.join((
            'module load MDTools;',
            'charmm2lammps.pl all36_lipid_extended_stripped {system_name:s}_psfgen',
            '-border=0 -lx={box[0]:.3f} -ly={box[1]:.3f} -lz={box[2]:.3f} ' ))

        # lammps
        self._template_lmp_cmd = lmp_cmd = ' '.join((
            'module load LAMMPS;',
            'mpirun ${{MPIRUN_OPTIONS}} lmp -in {inputFile:s}' ))

        # VMD
        self._template_vmd_cmd = ' '.join((
            'module load VMD;',
            'vmd -eofexit -e {infile:s}' ))

        # image conversion
        self._template_convert_cmd = 'convert "{infile:s}" "{outfile:s}"'

        self._template_pizzapy_merge_cmd = ' '.join((
            'module load MDTools/jlh-25Jan19-python-2.7;', # pizza.py contained her
            'pizza.py -f merge.py "{datafile:s}" "{reffile:s}" "{outfile:s}"' ))

        # netcdf2data.py
        self._template_netcdf2data_cmd = ' '.join((
            'module load MDTools Ovito;',
            'netcdf2data.py {datafile:s} {trajfile:s}'))

    def use_nemo_templates(self):
        # TODO: find way to wrap environment elegantly elsewhere
        self._template_sb_replicate_cmd = '{template_prefix:s}' + os.sep \
            + "replicate.sh {multiples[0]:d} {multiples[1]:d} " \
            + "{multiples[2]:d} {plane:03d}"

        self._template_packmol_cmd = ' '.join((
            'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
            'module use ${{HOME}}/modulefiles;',
            'module load mdtools;',
            'packmol < {infile:s}' ))

        self._template_packmol2gmx_cmd  = '{template_prefix:s}' \
            + os.sep + 'pdb_packmol2gmx.sh {infile:s}' # make path-indep

        self._template_gmx2pdb_cmd = ' '.join((
            'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
            'module use ${{HOME}}/modulefiles;',
            'module load gromacs/2018.1-gnu-5.2;',
            'bash {infile:s}' ))

        # psfgen
        self._template_psfgen_cmd  = ' '.join((
            'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
            'module use ${{HOME}}/modulefiles;',
            'module load vmd/1.9.3-text;',
            'vmd -e {infile:s}' ))

        # charmm2lammps.pl
        self._template_ch2lmp_cmd  = ' '.join((
            'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
            'module use ${{HOME}}/modulefiles;',
            'module load mdtools;',
            'charmm2lammps.pl all36_lipid_extended_stripped {system_name:s}_psfgen',
            '-border=0 -lx={box[0]:.3f} -ly={box[1]:.3f} -lz={box[2]:.3f} ' ))

        # lammps
        self._template_lmp_cmd = lmp_cmd = ' '.join((
            'module purge;',
            'module load lammps/16Mar18-gnu-7.3-openmpi-3.1-colvars-09Feb19;',
            'mpirun ${{MPIRUN_OPTIONS}} lmp -in {inputFile:s}' ))

        # VMD
        self._template_vmd_cmd = ' '.join((
            'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
            'module use ${{HOME}}/modulefiles;',
            'module load vmd/1.9.3-text;',
            'vmd -eofexit -e {infile:s}' ))

        # image conversion
        self._template_convert_cmd = 'convert "{infile:s}" "{outfile:s}"'

        #self._template_pizzapy_merge_cmd = None # not set up on NEMO yet
        self._template_pizzapy_merge_cmd = ' '.join((
            'module use ${{HOME}}/modulefiles;',
            'type deactivate && deactivate;', # deactivate virtual env if loaded
            'module --force purge;',
            'module load devel/python/2.7.14 mpi/openmpi/2.1-gnu-4.8;',
            'source ${{HOME}}/venv/jlh-nemo-python-2.7/bin/activate;',
            'module load jlh/mdtools/26Jun18-jlh-python-2.7;', # pizza.py contained her
            'pizza.py -f merge.py "{datafile:s}" "{reffile:s}" "{outfile:s}"' ))

        # netcdf2data.py
        self._template_netcdf2data_cmd = ' '.join((
            'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
            'module use ${{HOME}}/modulefiles;',
            'module load ovito/3.0.0-dev234;',
            'module load jlh/mdtools/26Jun18-jlh;',
            'netcdf2data.py {datafile:s} {trajfile:s}'))

    def use_juwels_templates(self):
        # LAMMPS
        self._template_lmp_cmd = lmp_cmd = ' '.join((
            'module use /gpfs/homea/hka18/hka184/modules/modulefiles;',
            'module load jlh/lammps/16Mar18-intel;',
            'srun lmp -in {inputFile:s}' ))

        # VMD
        self._template_vmd_cmd = ' '.join((
            'module use ${{HOME}}/modules/modulefiles;',
            'module load jlh/vmd/1.9.3-text;',
            'vmd -eofexit -e {infile:s}' ))

        # image conversion
        self._template_convert_cmd = ' '.join((
            'module --force purge;',
            'module load GCCcore/.5.5.0 ImageMagick/6.9.9-19;',
            'convert "{infile:s}" "{outfile:s}"' ))

        # pizza-py merge datafiles
        self._template_pizzapy_merge_cmd = ' '.join((
            'module use ${{HOME}}/modules/modulefiles;',
            'type deactivate && deactivate;', # deactivate virtual env if loaded
            'module --force purge;',
            'module load GCC/5.5.0 Python/2.7.14;',
            'source ${{HOME}}/venv/jlh-juwels-python-2.7/bin/activate;',
            'module load jlh/mdtools/26Jun18-jlh-python-2.7;', # pizza.py contained her
            'pizza.py -f merge.py "{datafile:s}" "{reffile:s}" "{outfile:s}"' ))

    def _set_machine_independent_templates(self):
        self._template_geninfo = \
            "Generated on {datetime:s} by {user:s}@{machine:s}"

        self._template_substrate_name = \
            "AU_{plane:03d}_{multiples[0]:d}x{multiples[1]:d}x{multiples[2]:d}"

        self._sb_replicate_suffix = '_sb_replicate'
        self._packmol_suffix      = '_packmol'
        self._packmol2gmx_suffix  = '_packmol2gmx'
        self._gmx_solvate_suffix  = '_gmx_solvate'
        self._gmx2pdb_suffix      = '_gmx2pdb'
        self._psfgen_suffix       = '_psfgen'
        self._ch2lmp_suffix       = '_ch2lmp'

    # 96 tasks per node is standard setting on JUWELS
    def use_bwcloud_queue( self,
        packmol_queueadapter = {},
        minimization_queueadapter = {},
        equilibration_queueadapter = {},
        production_queueadapter   = {},
        std_worker          = 'bwcloud_std',
        queue_worker        = 'bwcloud_std',
        queue_worker_serial = 'bwcloud_std'):

        self.std_worker         = std_worker
        self.queue_worker       = queue_worker
        self.queue_worker_serial= queue_worker_serial
        self.packmol_queueadapter      = packmol_queueadapter
        self.minimization_queueadapter = minimization_queueadapter
        self.equilibration_queueadapter = equilibration_queueadapter
        self.production_queueadapter   = production_queueadapter

    # NEMO std settings 20 ppn
    def use_nemo_queue( self,
        packmol_queueadapter = {
            'walltime' : '24:00:00',
            'nodes':       1,
            'ppn':         1,
        },
        minimization_queueadapter = {
            'walltime' : '04:00:00',
            'nodes':       1
        },
        equilibration_queueadapter = {
            'walltime' : '24:00:00',
            'nodes':       2,
        },
        production_queueadapter = {
            'walltime' : '96:00:00',
            'nodes':       4,
        },
        std_worker          = 'nemo_noqueue',
        queue_worker        = 'nemo_queue',
        queue_worker_serial = 'nemo_queue_serial'):

        self.std_worker         = std_worker
        self.queue_worker       = queue_worker
        self.queue_worker_serial= queue_worker_serial
        self.packmol_queueadapter      = packmol_queueadapter
        self.minimization_queueadapter = minimization_queueadapter
        self.equilibration_queueadapter = equilibration_queueadapter
        self.production_queueadapter   = production_queueadapter

    # 96 tasks per node is standard setting on JUWELS
    def use_juwels_queue( self,
        # nocont walltime 6h
        packmol_queueadapter = {
            'walltime' : '06:00:00',
            'ntasks':      1,
        },
        minimization_queueadapter = {
            'walltime' : '06:00:00',
            'ntasks':    96
        },
        equilibration_queueadapter = {
            'walltime' : '06:00:00',
            'ntasks':     96
        },
        production_queueadapter   = {
            'walltime' : '06:00:00',
            'ntasks'   : 192
        },
        std_worker          = 'juwels_std',
        queue_worker        = 'juwels_queue',
        queue_worker_serial = 'juwels_queue_serial'):

        self.std_worker         = std_worker
        self.queue_worker       = queue_worker
        self.queue_worker_serial= queue_worker_serial
        self.packmol_queueadapter      = packmol_queueadapter
        self.minimization_queueadapter = minimization_queueadapter
        self.equilibration_queueadapter = equilibration_queueadapter
        self.production_queueadapter   = production_queueadapter

    def geninfo(self, reset=False, static_str=None):
        if static_str is not None:
            self._geninfo = static_str
        elif reset or (self._geninfo is None):
            user = os.getlogin()
            machine = os.uname().nodename
            self._geninfo = self._template_geninfo.format(
                datetime=datetime.datetime.now().ctime(),
                user=user,
                machine=machine)
        return self._geninfo

# Queries

    def identifyIncompleteWorkflows(self,
        precedent_step_name='packmol', subsequent_step_name='ch2lmp'):
        """Expects name of two (not necessarily directly) subsequent steps,
        and returns a dict of dict, identifying Firework IDs of
        (healthy, i.e. non-fizzled) preceding steps that do not have
        healthy subsequent steps.
        Returns:
        --------
          {str: {int: [int] } dict of dict of list, i.e.
              { system_name: { workflow_id: [firework_id] }
        """

        # get ALL healthy "preceding step" fireworks:
        preceding_step_dict = self.query_step(step=precedent_step_name,
            state_list = ['COMPLETED','RUNNING','RESERVED','READY','WAITING'])

        # system_name, wf_id : list(fw_id) dict
        preceding_step_wf_dict_dict = {}
        # system_name : set(wf_id) dict
        preceding_step_wf_set_dict = {}
        # go through ALL found fireworks and "group by" workflow
        # workflows are indentified by their "zero-eth" root firework
        for system_name, fw_id_lst in preceding_step_dict.items():
            root_fw_id_lst = []
            preceding_step_wf_dict_dict[system_name] = {}
            for fw_id in fw_id_lst:
                # identify workflow uniquely by first root fw id
                wf_id = self.lpad.get_wf_by_fw_id( fw_id ).root_fw_ids[0]
                root_fw_id_lst.append( wf_id )
                if wf_id in preceding_step_wf_dict_dict[system_name]:
                    preceding_step_wf_dict_dict[system_name][wf_id].append(fw_id)
                else:
                    preceding_step_wf_dict_dict[system_name][wf_id] = [fw_id]
            preceding_step_wf_set_dict[system_name] = set( root_fw_id_lst )

        # get ALL healthy "subsequent step" fireworks:
        subsequent_step_dict = self.query_step(step=subsequent_step_name,
            state_list = ['COMPLETED','RUNNING','RESERVED','READY','WAITING'])

        # system_name, wf_id : list(fw_id) dict
        subsequent_step_wf_dict_dict = {}
        # system_name : set(wf_id) dict
        subsequent_step_wf_set_dict = {}
        # again group by workgflow:
        for system_name, fw_id_lst in subsequent_step_dict.items():
            root_fw_id_lst = []
            subsequent_step_wf_dict_dict[system_name] = {}
            for fw_id in fw_id_lst:
                # identify workflow uniquely by first root fw id
                wf_id = self.lpad.get_wf_by_fw_id( fw_id ).root_fw_ids[0]
                root_fw_id_lst.append( wf_id )
                if wf_id in subsequent_step_wf_dict_dict[system_name]:
                    subsequent_step_wf_dict_dict[system_name][wf_id].append(fw_id)
                else:
                    subsequent_step_wf_dict_dict[system_name][wf_id] = [fw_id]

            # this set dict just holds sets of all workflows (identified by
            # root fw_id) containing at least one healthy subsequent step
            # for the system identified by "key"
            subsequent_step_wf_set_dict[system_name] = set( root_fw_id_lst )

        # first filter all workflows that have at least one healthy firework
        # of the "precedent" step, but none of the "subsequent" step:

        # compare keys (the case of "subsequent steps" without the according
        # "preceding step" is assumed not to occur)
        non_existent_subsequent = preceding_step_wf_set_dict.keys() - subsequent_step_wf_set_dict.keys()

        # treatment of systems non-existent in consecutive step
        wf_discrepancy_dict = {}
        if non_existent_subsequent: #non-empty:
            logging.warn(
                "{:d} systems not present at all in subsequent step.".format(
                    len(non_existent_subsequent)))
            wf_discrepancy_dict = {
                system_name: preceding_step_wf_set_dict[system_name] \
                for system_name in non_existent_subsequent }

        # next look at systems, where at least one workflow exists holding both
        # healthy preceding and subsequent step
        system_overlap = preceding_step_wf_set_dict.keys() & subsequent_step_wf_set_dict.keys()
        logging.info(
            "{:d} systems both present in preceding and subsequent step.".format(
                len(system_overlap)))
        # within those sets, identify the workflows not holding at least one
        # healthy subsequent step
        for system_name in system_overlap:
            diff = preceding_step_wf_set_dict[system_name] - subsequent_step_wf_set_dict[system_name]
            if diff:
                logging.info( ' '.join((
                    "System {:s} has {:d} systems".format(system_name, len(diff)),
                    "present in preceding step, but not in subsequent step")) )
                wf_discrepancy_dict[system_name] = diff

        # get fireworks ids within those workflows
        fw_discrepancy_dict = {}
        for system_name, wf_id_set in wf_discrepancy_dict.items():
            fw_discrepancy_dict[system_name] = {}

            for wf_id in wf_id_set:
                fw_discrepancy_dict[system_name][wf_id] = preceding_step_wf_dict_dict[system_name][wf_id]

        # TODO
        # ATTENTION: so far the "precedent" - "subsequent" naming is misleading
        # it is very well possible that a workflow contains forked branches
        # with both healthy preceding and subsequent steps in one branch, but
        # only healthy preceding and no (or unhealthy) subsequent step
        # in the other. This workflow (and the latter preceding firework) will
        # not be identified.
        return fw_discrepancy_dict

    def identifyIncompleteWorkflowsInScope(self,
        precedent_step_name, subsequent_step_name, system_name_scope):
        fw_dict = self.identifyIncompleteWorkflows(
            precedent_step_name, subsequent_step_name)
        # key is system_name, values are wf_id: [ fw_id ] dicts
        in_scope_dict = { key: fw_dict[key] for key in
                          fw_dict.keys() & system_name_scope }
        # flatten second dimension of in_scope_dict (remove wf ids)
        in_scope_dict = { system_name: [
            fw_id for fw_id_lst in fw_id_lst_dict.values() for fw_id in fw_id_lst ]
            for system_name, fw_id_lst_dict in in_scope_dict.items() }
        return in_scope_dict

    def query_step( self,
                    step = 'prepare_system_files',
                    state_list = ['COMPLETED', 'RUNNING', 'RESERVED', 'READY',
                        'WAITING'],
                    group_key = 'system_name',
                    step_key  = 'step'):
        '''queries spec.step_key, groups by system_name, returns dict'''

        fw_id_list = self.lpad.get_fw_ids(
            query = {
                'spec.{}'.format(step_key):  step,
                'state':                     { '$in': state_list },
                'spec.{}'.format(group_key): { '$exists': True }
            },
            sort = [('updated_on', pymongo.ASCENDING)] )

        logging.info("fw_id_list: {}".format(fw_id_list))

        system_name_list = [
            self.lpad.get_fw_dict_by_id(fw_id)["spec"][group_key] \
                for fw_id in fw_id_list ]

        system_name_set = set(system_name_list)
        system_names = list(system_name_set)

        fw_id_dict = {}
        for system_name in system_names:
            fw_id_dict[system_name] = self.lpad.get_fw_ids(
            query = {
                'spec.{}'.format(step_key):  step,
                'spec.{}'.format(group_key): system_name,
                'state':                     { '$in': state_list }
            },
            sort = [('updated_on', pymongo.ASCENDING)] )

        return fw_id_dict

    def _update_query(self, query, key, obj):
        """ query: dict of obj, key: str, obj: int, str, dict or list """
        if obj is not None:
            # add allowed types here:
            if (type(obj) is int) or (type(obj) is str) or (type(obj) is dict):
                query.update( {key: obj} )
            elif (type(obj) is list):
                query.update( {key: { '$in': obj }} )
            else:
                raise ValueError( ' '.join((
                    "Value of {} must be int, str, dict or list,".format(key),
                    "but is {}".format(type(obj)) )) )

    def query_systems( self, system_names,
        step = None, state = None, restart=None ):
        """Returns a dictionary str : list of int  with each system's
        Firework ids sorted ascending by update time (latest last)"""

        fw_id_list = []
        for system_name in system_names:
            # earlier inconsistent keys
            query = { '$or': [
                {'spec.system_name':   system_name },
                {'spec.system':        system_name } ]
            }
            self._update_query(query, 'state',        state )
            self._update_query(query, 'spec.step',    step )
            self._update_query(query, 'spec.restart', restart )

            logging.info( '{:s} query: {}'.format( system_name, query ) )

            fw_id_list.append(
                self.lpad.get_fw_ids(
                    query = query,
                    sort = [('updated_on', pymongo.ASCENDING)] ) )

            fw_id_dict = dict( zip( system_names, fw_id_list ) )
        return fw_id_dict

    def get_set_of_prepared_morphologies( self ):
        # eayrlier, used key 'system' instead of 'system_name' earlier
        fw_id_dict = self.query_step( 'ch2lmp', state_list=['COMPLETED'],
            group_key='system_name' )
        return set( fw_id_dict.keys() )

    def get_set_of_healthy_morphologies( self ):
        fw_id_dict = self.query_step( 'ch2lmp', state_list=[
            'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ])
        return set( fw_id_dict.keys() )

    def get_set_of_prepared_minimizations( self ):
        fw_id_dict = self.query_step( 'prepare_system_files',
            state_list=['COMPLETED'] )
        return set( fw_id_dict.keys() )

    def get_set_of_healthy_minimization_preparations( self ):
        fw_id_dict = self.query_step( 'prepare_system_files', state_list=[
            'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ])
        return set( fw_id_dict.keys() )

    def get_set_of_minimized_systems( self ):
        fw_id_dict = self.query_step( 'minimization',
            state_list = ['COMPLETED'] )
        return set( fw_id_dict.keys() )

    def get_set_of_healthy_minimizations( self ):
        fw_id_dict = self.query_step( 'minimization', state_list=[
            'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ])
        return set( fw_id_dict.keys() )

    def get_set_of_nvt_equilibrated_systems( self  ):
        fw_id_dict = self.query_step( 'equilibration_nvt',
            state_list = ['COMPLETED'] )
        return set( fw_id_dict.keys() )

    def get_set_of_healthy_nvt_equilibrations( self ):
        fw_id_dict = self.query_step( 'equilibration_nvt', state_list=[
            'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ])
        return set( fw_id_dict.keys() )

    def get_set_of_npt_equilibrated_systems( self ):
        fw_id_dict = self.query_step( 'equilibration_npt',
            state_list = ['COMPLETED'] )
        return set( fw_id_dict.keys() )

    def get_set_of_healthy_npt_equilibrations( self ):
        fw_id_dict = self.query_step( 'equilibration_npt', state_list=[
            'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ])
        return set( fw_id_dict.keys() )

    def get_set_of_evolved_systems( self ):
        production_10ns_systems_dict = self.query_step(
            '10ns_production_mixed', state_list = [ 'COMPLETED' ] )
        return set( production_10ns_systems_dict.keys() )

    def get_set_of_fizzled_production_runs( self ):
        fizzled_production_10ns_systems_dict = self.query_step(
            '10ns_production_mixed',
            state_list=[ 'FIZZLED' ])
        return set( fizzled_production_10ns_systems_dict.keys() )

    def get_set_of_healthy_production_runs( self ):
        healthy_production_10ns_systems_dict = self.query_step(
            '10ns_production_mixed', state_list = [ 'COMPLETED', 'RESERVED',
            'RUNNING', 'READY', 'WAITING' ])
        return set( healthy_production_10ns_systems_dict.keys() )

    # set operations
    def get_to_prepare( self ):
        systems_to_prep =  self.get_set_of_prepared_morphologies() \
            - self.get_set_of_healthy_minimization_preparations()
        return list(systems_to_prep)

    def get_to_minimize( self ):
        systems_to_prep = self.get_set_of_healthy_minimization_preparations() \
            - self.get_set_of_healthy_minimizations()
        return list(systems_to_prep)

    def get_to_nvt_equlibrate( self ):
        systems_to_prep = self.get_set_of_healthy_minimizations() \
            - self.get_set_of_healthy_nvt_equilibrations()
        return list(systems_to_prep)

    def get_to_npt_equlibrate( self ):
        systems_to_prep = self.get_set_of_healthy_nvt_equilibrations() \
            - self.get_set_of_healthy_npt_equilibrations()
        return list(systems_to_prep)

    def get_to_run_production( self ):
        systems_to_prep = self.get_set_of_healthy_npt_equilibrations() \
            - self.get_set_of_healthy_production_runs()
        return list(systems_to_prep)

    # TODO: rework, does not work anymore with current framework
    # with automatic requeueing after expired walltime
    def get_discontinued_production_runs( self ):
        # get all fizzled production Fireworks
        fizzled_production_10ns_systems_dict = self.query_step(
            step = '10ns_production_mixed',
            state_list = [ 'FIZZLED' ] )

        # query all production workflows not archived
        fizzled_production_wf_dict = {}
        for key, fw_ids in fizzled_production_10ns_systems_dict.items():
            fizzled_production_wf_dict[key] = self.lpad.get_wf_ids(
                query = {
                    'name' : key + '_prepare_system_files',
                    'state': { '$ne': 'ARCHIVED' } } )

        # get all leaves of the wfs queried above
        fizzled_production_wf_leaf_fw_dict = {}
        for key, wf_id in fizzled_production_wf_dict.items():
            if len(wf_id) > 0:
                wf = self.lpad.get_wf_by_fw_id(wf_id[-1])
                fizzled_production_wf_leaf_fw_dict[key] = wf.leaf_fw_ids

        # intersection of fizzled fw ids and wf leaf ids
        # (check which fizzled production fw are leaves)
        fizzled_production_10ns_leaves_dict = {}
        for key, fw_ids in fizzled_production_10ns_systems_dict.items():
            if key in fizzled_production_wf_leaf_fw_dict:
                intersection = \
                    set(fizzled_production_wf_leaf_fw_dict[key]) & set(fw_ids)
                if intersection != set():
                    fizzled_production_10ns_leaves_dict[key] = intersection
        return fizzled_production_10ns_leaves_dict

    # latest last in list
    def get_healthy( self, system_names,step='minimization' ):
        fw_id_dict = { system_name: self.lpad.get_fw_ids(
            query = {
                'spec.step':          step,
                'state':              {
                    '$in': [
                        'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ]
                    },
                'spec.system_name':   system_name
            },
            sort = [('updated_on', pymongo.ASCENDING)] ) \
            for system_name in system_names }
        return fw_id

# Launches
    def append_wf_by_key( self, new_wf_dict, prev_fw_id_dict,add_if_missing=False):
        """Expects a dictionary of new work flows and a dictionary of
        previously existent fireworks on launchpad. Appends by key."""
        for key, wf in new_wf_dict.items():
            if prev_fw_id_dict[key] is None or prev_fw_id_dict[key] == []:
                print("{} empty in prev_fw_id_dict".format(key))
                if add_if_missing:
                    self.lpad.add_wf(wf)
                    print("Added.")
                else:
                    print("Skipped.")
                    continue

            print("Append {} to last fw_id in list {}".format(
                key, prev_fw_id_dict[key]))

            if type(wf) is Firework:
                wf = Workflow([wf])

            self.lpad.append_wf(wf,[ prev_fw_id_dict[key][-1]])

    def append_fw_to_step( self, fw_dict, step='minimization' ):
        fw_id_dict = self.get_healthy(
            system_names = list( fw_dict.keys() ), step = step )
        for system_name, fw in fw_dict.items():
            print( ' '.join(( "{}: ".format(system_name),
                "Append a new Firework to previous Firework",
                "of last ID in list {}".format(fw_id_dict[system_name]) )) )
            self.lpad.append_wf( Workflow([fw]) ,[fw_id_dict[system_name][-1]] )

    def add_fw( self, fw_dict ):
        fw_id_dict = { system_name: self.lpad.add_wf( Workflow( fw ) ) \
                for system_name, fw in fw_dict.items() }
        return fw_id_dict

    def add_wf( self, wf_dict ):
        fw_id_dict = { system_name: self.lpad.add_wf( wf ) \
                for system_name, wf in wf_dict.items() }
        return fw_id_dict

# Administration

    def defuse_children(self, fw_id, wf = None, dry_run = False):
        """Descends to children recursively and defuses them
        (including this parent fw_id)
        Parameters
        ----------
        fw_id: int
        wf : fireworks.Workflow
            will be determined automatically
        dry_run : bool
            Only descends and logs, does not defuse if True
        Returns
        -------
        list of int
            all defused fw_id
        """
        if wf is None:
            wf = self.lpad.get_wf_by_fw_id(fw_id)

        defused_fw_id_list = []
        if fw_id not in wf.links:
            logging.warn("fw_id {} not in wf.links!".format(fw_id))
        else:
            for child_fw_id in wf.links[fw_id]:
                defused_fw_id_list.extend( self.defuse_children( child_fw_id ) )

        if not dry_run:
            self.lpad.defuse_fw(fw_id, rerun_duplicates=False)

        defused_fw_id_list.append(fw_id)
        logging.info("fw_id {} defused".format(fw_id))
        return defused_fw_id_list

# Workflows, Fireworks & Firetasks

# Scripted preassembly of surfactant aggregates at the AU 111 surface.
#
# Requires
#
# * modified CHARMM36 Jul17 package https://github.com/jotelha/jlh_toppar_c36_jul17
# * modified GROMACS 2018.1 top folder including charmm36.ff https://github.com/jotelha/jlh_gmx_2018.1_top
# * modified pdb-tools forked from https://github.com/haddocking/pdb-tools
# * GROMACS 2018.1
# * vmd-1.9.3 with psfgen plugin, e.g. pre-compiled distribution vmd-1.9.3.bin.LINUXAMD64.text.tar.gz
#   from http://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD
# * LAMMPS 16Mar18
# * charmm2lammps (comes with LAMMPS 16Mar18 sources)
#
# 1. Create GROMACS .hdb, .rtp for surfactants, dummy .rtp for ions
# 2. Load system description from some stored pandas.Dataframe, e.g. from pickle.
#    Should contain information on substrate measures, box size, etc.
#    ADD DETAILED DESCRIPTION OF EXPECTED VALUES.
# 3. Identify all unique substrate slabs and create according .pdb files from multiples of unit cell with
#    `gmx genconf`.
# 4. Create a subfolter for every system and copy (or links) some necessary files
# 5. Use packmol to create bilayer on gold surface
# 6. Use `pdb_packmol2gmx` to run sum simple renumbering on residues.
#    This is mainly necessary because gmx needs each substrate atom to have its own residue number.
# 7. Create GROMACS .gro and .top from .pdb, solvate (and ionize system).
#    A bash script suffixed `_gmx_solvate.sh` is created for this purpose.
# 8. Split .gro system into .pdb chunks of at most 9999 residues
#    in order not to violate .pdb format restritcions. psfgen cannot work on .pdb > 9999 residues.
#    The bash script suffixed `_gmx2pdb.sh` does this.
# 9. Generate .psf with VMD's `psfgen`.
# 10.Generate LAMMPS data from psgen-generated .psf and .pdb with `charmm2lammps.pl`
#
# TODO: Doe solvation and ionization in packmol as well, remove the detour via GROMACS


# substrates (on NEMO), to be tested
    def prepare_substrates(self, system_names=None):
        """Processes attribute _sim_df and constructs substrate slabs
        _sim_df should have the columns
          sb_crystal_plane
          sb_multiples
        A list of system_names can be used to only process a subset."""

        if system_names is None:
            sim_df = self._sim_df
        else:
            sim_df = self._sim_df.loc[system_names]

        unique_sb_crystal_planes = sim_df["sb_crystal_plane"].unique()

        sb_names = []
        sb_replicate_fw_list = []
        for plane in unique_sb_crystal_planes:
            unique_sb_multiples = np.unique(
                   np.vstack( sim_df[
                       sim_df["sb_crystal_plane"] == plane
                   ][["sb_multiples"]].values.flatten() ) ,
                   axis=0 )
            for multiples in unique_sb_multiples:
                sb_name = self._template_substrate_name.format(
                    plane=plane, multiples=multiples )
                sb_names.append(sb_name)

                # create all substrate files
                sb_get_unit_cell_ft = FileTransferTask( {
                    'files': [ {
                        'src':  self.template_prefix + os.sep \
                            + 'au_cell_P1_111.gro', # allow other surfaces here
                        'dest': 'au_cell_P1_111.gro' } ],
                    'mode': 'copy' } )

                sb_replicate_ft =  ScriptTask.from_str(
                    self._template_sb_replicate_cmd.format(
                        template_prefix = self.template_prefix,
                        plane           = plane,
                        multiples       = multiples ) , {
                    'stdout_file':  sb_name + self._sb_replicate_suffix + '.out',
                    'stderr_file':  sb_name + self._sb_replicate_suffix + '.err',
                    'use_shell':    True,
                    'fizzle_bad_rc':True } )

                # Attention: Explicitly overwrite previously stored file in DB:
                sb_delete_ft = DeleteFilesTask( {
                    'identifiers': ["{:s}.pdb".format(sb_name)]})
                sb_store_ft =  AddFilesTask( {
                    'paths':       ["{:s}_tidy.pdb".format(sb_name)],
                    'identifiers': ["{:s}.pdb".format(sb_name)]})

                sb_replicate_fw = Firework( [
                        sb_get_unit_cell_ft,
                        sb_replicate_ft,
			sb_delete_ft,
                        sb_store_ft
                    ],
                    spec={
                        "_category":   self.std_worker,
                        "_dupefinder": DupeFinderExact(),
                        "substrate":   sb_name,
                        "step"     :   "sb_replicate",
                        "geninfo"  :   self.geninfo()
                    },
                name="{:s}_sb_replicate".format(sb_name) )

                sb_replicate_fw_list.append( [ sb_replicate_fw ] )

                #sb_replicate_wf = Workflow( [sb_replicate_fw],
                #    name="{:s}_sb_replicate".format(sb_name) )

                # sb_replicate_wf_list.append( sb_replicate_wf )

        sb_name_fw_dict = dict( zip( sb_names, sb_replicate_fw_list ) )
        #sb_name_wf_dict = dict(zip(sb_names,sb_replicate_wf_list))
        return sb_name_fw_dict

# generic packing
    def prepare_packmol(self, system_name, pack_aggregates,
                        nloop = None, maxit = None ):
        """Creates preassembled aggregates on surface"""

        #for system_name, row in sim_df.loc[system_names].iterrows():
        row = self._sim_df.loc[system_name,:]
        surfactant  = row["surfactant"]
        counterion  = row["counterion"]
        sfN         = row["sf_nmolecules"]

        sb_name     = row["sb_name"]

        # measures of substrate:
        sb_measures = np.asarray(row["sb_measures"]) / C.angstrom

        # standard settings can be overridden
        packmol_script_writer_task_context = {
            'header':        self.geninfo(),
            'system_name':   system_name,
            'sb_name':       sb_name,
            'tolerance':     tolerance,
            'write_restart': True
        }

        if nloop is not None:
            packmol_script_writer_task_context['nloop'] = nloop

        if maxit is not None:
            packmol_script_writer_task_context['maxit'] = maxit

        packmol_script_writer_task_context.update(
            pack_aggregates(
                surfactant  = surfactant,
                counterion  = counterion,
                sfN         = sfN,
                sb_measures = sb_measures,
            )
        )

        # packmol fill script template
        packmol_fill_script_template_ft = TemplateWriterTask( {
            'context' :      packmol_script_writer_task_context,
            'template_file': self.template_prefix + os.sep + 'packmol.inp',
            'output_file':   system_name + self._packmol_suffix + '.inp' } )

        packmol_fill_script_template_fw = Firework(
            packmol_fill_script_template_ft,
            spec={
                "_category":   self.std_worker,
                "_dupefinder": DupeFinderExact(),
                '_files_out': {
                    'packmol_inp': system_name + self._packmol_suffix + '.inp'
                    },
                "system_name": system_name,
                "step"   :     "packmol_fill_script_template",
                "geninfo":     self.geninfo() # serves to distinguish duplicates
            },
            name="{:s}_packmol_fill_script_template".format(system_name) )

    # packmol get building blocks
        single_surfactant_pdb = '1_{:s}.pdb'.format(surfactant)
        single_counterion_pdb = '1_{:s}.pdb'.format(counterion)

        packmol_get_components_ft = FileTransferTask( {
                'files': [ {
                    'src':  os.path.join(
                        self.template_prefix, single_surfactant_pdb ),
                    'dest': single_surfactant_pdb }, {
                    'src':  os.path.join(
                        self.template_prefix, single_counterion_pdb ),
                    'dest': single_counterion_pdb } ],
                'mode': 'copy' } )

        # load surface from data base
        packmol_get_substrate_ft = GetFilesTask( {
                'identifiers': [ '{:s}.pdb'.format(sb_name) ],
                'new_file_names': [ '{:s}.pdb'.format(sb_name) ] } )

    # packmol run
        packmol_cmd = self._template_packmol_cmd.format(
            infile= ( system_name + self._packmol_suffix + '.inp' ) )
        packmol_ft =  ScriptTask.from_str( packmol_cmd,{
            'stdout_file':  system_name + self._packmol_suffix + '.out',
            'stderr_file':  system_name + self._packmol_suffix + '.err',
            'use_shell':    True,
            'fizzle_bad_rc':True } )

        mkdir_ft =  ScriptTask.from_str(
            'mkdir -p "{:s}"'.format(
                self.output_prefix + os.sep + system_name ),
                {
                    'use_shell':    True, 'fizzle_bad_rc': False
                } )

        copy_output_files_ft = FileTransferTask( {
            'files': [
                system_name + self._packmol_suffix + '.inp',
                system_name + self._packmol_suffix + '.pdb',
            ],
            'dest': os.path.join(self.output_prefix,system_name),
            'mode': 'copy' } )

        packmol_fw = Firework(
            [
                packmol_get_components_ft,
                packmol_get_substrate_ft,
                packmol_ft,
                mkdir_ft,
                copy_output_files_ft
            ],
            spec={
                "_queueadapter": self.packmol_queueadapter,
                "_category":   self.queue_worker_serial,
                "_dupefinder": DupeFinderExact(),
                "_files_in" : {
                    "packmol_inp":  (system_name + self._packmol_suffix + '.inp') },
                "_files_out": {
                    "packmol_pdb" : (system_name + self._packmol_suffix + '.pdb'),
                    "packmol_pdb_FORCED" : (system_name + self._packmol_suffix + '.pdb_FORCED')
                },
                "system_name": system_name,
                "step"     :   "packmol",
                "geninfo":     self.geninfo() # serves to distinguish duplicates
            },
            name= system_name + self._packmol_suffix,
            parents=[packmol_fill_script_template_fw] )

        packmol_wf = Workflow(
            [
                packmol_fill_script_template_fw,
                packmol_fw
            ],
            {
                packmol_fill_script_template_fw: packmol_fw
            },
            name = system_name + self._packmol_suffix )

        return packmol_wf


# monolayer
    def pack_monolayer(self, sfN, sb_measures, surfactant, counterion,
                       l_surfactant     = l_SDS,
                       head_atom_number = head_atom_number_SDS,
                       tail_atom_number = tail_atom_number_SDS ):
        """Creates preassembled monolayer"""

        logging.info( "Monolayer with {:d} surfactant molecules.".format(sfN ) )

        sbX, sbY, sbZ = sb_measures

        # place box at coordinate zero in z-direction
        sb_pos = - sb_measures / 2 * np.array( [1,1,0] )

        # 1st monolayer above substrate, polar head towards surface
        # NOT applying http://www.ime.unicamp.br/~martinez/packmol/userguide.shtml
        # recommendation on periodic bc

        na_layer_1_bb  = np.array([ [ - sbX / 2.,
                                        sbX / 2. ],
                                    [ - sbY / 2.,
                                        sbY / 2. ],
                                    [ sbZ,
                                      sbZ + tolerance ] ])

        monolayer_bb_1 = np.array([ [ - sbX / 2.,
                                        sbX / 2. ],
                                    [ - sbY / 2.,
                                        sbY / 2. ],
                                    [ sbZ,
                                      sbZ + 2*tolerance + l_surfactant ] ] )

        lower_constraint_plane_1 = sbZ + 1*tolerance
        upper_constraint_plane_1 = sbZ + 1*tolerance + l_surfactant

        monolayers = [
            {
                'surfactant':             surfactant,
                'N':                      sfN,
                'lower_atom_number':      tail_atom_number,
                'upper_atom_number':      head_atom_number,
                'bb_lower':               monolayer_bb_1[:,0],
                'bb_upper':               monolayer_bb_1[:,1],
                'lower_constraint_plane': lower_constraint_plane_1,
                'upper_constraint_plane': upper_constraint_plane_1
            } ]
        ionlayers = [
            {
                'ion':                    counterion,
                'N':                      sfN,
                'bb_lower':               na_layer_1_bb[:,0],
                'bb_upper':               na_layer_1_bb[:,1]
            } ]

        context = {
            'sb_pos':     sb_pos,
            'monolayers': monolayers,
            'ionlayers':  ionlayers
        }
        return context

    def pack_bilayer(  self, sfN, sb_measures, surfactant, counterion,
                       l_surfactant     = l_SDS,
                       head_atom_number = head_atom_number_SDS,
                       tail_atom_number = tail_atom_number_SDS ):
        """Creates a single bilayer on substrate with couinterions at polar heads"""
        sbX, sbY, sbZ = sb_measures

        # place box at coordinate zero in z-direction
        sb_pos = - sb_measures / 2 * np.array( [1,1,0] )

        N_inner_monolayer = (sfN // 2) + (sfN % 2)
        N_outer_monolayer = sfN//2

        na_layer_1_bb  = np.array([ [ - sbX / 2. ,
                                        sbX / 2. ],
                                    [ - sbY / 2. ,
                                        sbY / 2. ],
                                    [ sbZ,
                                      sbZ + tolerance ] ])

        monolayer_bb_1 = np.array([ [ - sbX / 2. ,
                                        sbX / 2. ],
                                    [ - sbY / 2. ,
                                        sbY / 2.  ],
                                    [ sbZ,
                                      sbZ + 2*tolerance + l_surfactant ] ])

        lower_constraint_plane_1 = sbZ + 1*tolerance
        upper_constraint_plane_1 = sbZ + 1*tolerance + l_surfactant
        z_shift_monolayer_2 = 1*tolerance + l_surfactant # overlap
        z_shift_na_layer_2 =  2*z_shift_monolayer_2 - 1*tolerance

        monolayer_bb_2 = monolayer_bb_1 + np.array([[0,0],[0,0],
                                                    [z_shift_monolayer_2,
                                                     z_shift_monolayer_2]])
        lower_constraint_plane_2 = lower_constraint_plane_1 + z_shift_monolayer_2
        upper_constraint_plane_2 = upper_constraint_plane_1 + z_shift_monolayer_2

        na_layer_2_bb = na_layer_1_bb + np.array([[0,0],[0,0],
                                                  [z_shift_na_layer_2,
                                                   z_shift_na_layer_2]])

        monolayers = [
            {
                'surfactant':             surfactant,
                'N':                      N_inner_monolayer,
                'lower_atom_number':      head_atom_number,
                'upper_atom_number':      tail_atom_number,
                'bb_lower':               monolayer_bb_1[:,0],
                'bb_upper':               monolayer_bb_1[:,1],
                'lower_constraint_plane': lower_constraint_plane_1,
                'upper_constraint_plane': upper_constraint_plane_1
            },
            {
                'surfactant':             surfactant,
                'N':                      N_outer_monolayer,
                'lower_atom_number':      tail_atom_number,
                'upper_atom_number':      head_atom_number,
                'bb_lower':               monolayer_bb_2[:,0],
                'bb_upper':               monolayer_bb_2[:,1],
                'lower_constraint_plane': lower_constraint_plane_2,
                'upper_constraint_plane': upper_constraint_plane_2
            } ]
        ionlayers = [
            {
                'ion':                    counterion,
                'N':                      N_inner_monolayer,
                'bb_lower':               na_layer_1_bb[:,0],
                'bb_upper':               na_layer_1_bb[:,1]
            },
            {
                'ion':                    counterion,
                'N':                      N_outer_monolayer,
                'bb_lower':               na_layer_2_bb[:,0],
                'bb_upper':               na_layer_2_bb[:,1]
            } ]

        context = {
            'sb_pos':     sb_pos,
            'monolayers': monolayers,
            'ionlayers':  ionlayers
        }
        return context

    def pack_cylinders(self, sfN, sb_measures, surfactant, counterion,
                       l_surfactant     = l_SDS,
                       head_atom_number = head_atom_number_SDS,
                       tail_atom_number = tail_atom_number_SDS,
                       ncylinders       = 3,
                       hemicylinders    = False ):
        """Creates preassembled (hemi-) cylinders on substrate with couinterions at polar heads"""

        hemistr = 'hemi-' if hemicylinders else ''
        logging.info( "{:d} {}cylinders with {:d} surfactant molecules in total.".format(ncylinders, hemistr, sfN ) )

        sbX, sbY, sbZ = sb_measures

        # place box at coordinate zero in z-direction
        sb_pos = - sb_measures / 2 * np.array( [1,1,0] )

        sf_molecules_per_cylinder = sfN // ncylinders
        excess_sf_molecules = sfN % ncylinders

        cylinder_spacing = sbY / ncylinders

        # cylinders parallelt to x-axis
        cylinders = [{} for _ in range(ncylinders)]
        ioncylinders = [{} for _ in range(ncylinders)]

        # surfactant cylinders
        #   inner constraint radius: 1*tolerance
        #   outer constraint radius: 1*tolerance + l_surfactant
        # ions between cylindric planes at
        #   inner radius:            1*tolerance + l_surfactant
        #   outer radius:            2*tolerance + l_surfactant
        for n, cylinder in enumerate(cylinders):
            cylinder["surfactant"] = surfactant

            if hemicylinders:
                cylinder["upper_hemi"] = True

            cylinder["inner_atom_number"] = tail_atom_number
            cylinder["outer_atom_number"] = head_atom_number

            cylinder["N"] = sf_molecules_per_cylinder
            if n < excess_sf_molecules:
                cylinder["N"] += 1

            # if packing hemicylinders, center just at substrate
            cylinder["base_center"] = [
                sb_pos[0],
                sb_pos[1] + (0.5 + float(n))*cylinder_spacing,
                sb_measures[2] ]
            # if packing full cylinders, shift center by one radius in z dir
            if not hemicylinders:
                cylinder["base_center"][2] += l_surfactant + 2*tolerance

            cylinder["length"] = sb_measures[0] - tolerance # to be on top of gold surfface
            cylinder["radius"] = 0.5*cylinder_spacing

            cylinder["inner_constraint_radius"] = tolerance

            maximum_constraint_radius = (0.5*cylinder_spacing - tolerance)
            cylinder["outer_constraint_radius"] = tolerance + l_surfactant \
                if tolerance + l_surfactant < maximum_constraint_radius \
                else maximum_constraint_radius

            logging.info(
                "Cylinder {:d} with {:d} molecules at {}, length {}, radius {}".format(
                n, cylinder["N"], cylinder["base_center"],
                cylinder["length"], cylinder["radius"]))

            # ions at outer surface
            ioncylinders[n]["ion"] = counterion

            if hemicylinders:
                ioncylinders[n]["upper_hemi"] = True

            ioncylinders[n]["N"] = cylinder["N"]
            ioncylinders[n]["base_center"] = cylinder["base_center"]
            ioncylinders[n]["length"] = cylinder["length"]
            ioncylinders[n]["inner_radius"] = cylinder["outer_constraint_radius"]
            ioncylinders[n]["outer_radius"] = cylinder["outer_constraint_radius"] + tolerance


        # experience shows: movebadrandom advantegous for (hemi-) cylinders
        context = {
            'sb_pos':        sb_pos,
            'cylinders':     cylinders,
            'ioncylinders':  ioncylinders,
            'movebadrandom': True,
        }
        return context

# convert pdb to lammps data format
    def prepare_pdb2lmp(self,system_name):
        """Processes _sim_df and constructs preassembled surfactant aggregates
        _sim_df should have the columns
          surfactant    -- the surfactant's name
          sf_nmolecules -- the number of surfactant molecules in the system
          box           -- measures of simulation box in SI units (m)
          sb_name       -- substrate slab name
          substrate     -- subsrtate material, i.e. AU
          counterion    -- counterion, i.e. NA or BR
          solvent       -- i.e. H2O or water
          sf_preassembly-- i.e. hemycylinders
          ci_preassembly-- i.e. at_polar_head
          sb_crystal_plane i.e. 111
        """

        row = self._sim_df.loc[system_name,:]

        sb_name    = row["sb_name"]

        surfactant = row["surfactant"]
        sfN        = row["sf_nmolecules"]

        # make more flexible at some point
        if surfactant == 'SDS': # surfactant is anionic
            nanion = 0
            ncation = sfN
        else: # CTAB (surfactant is cationic)
            nanion = sfN
            ncation = 0

        box_nanometer = np.asarray( row["box"] ) / C.nano
        box_angstrom  = np.asarray( row["box"] ) / C.angstrom

        consecutive_fw_list = []

    # retrieve packmol output file, even if fizzled (probably due to walltime)

        mkdir_ft =  ScriptTask.from_str(
            'mkdir -p "{:s}"'.format(
                self.output_prefix + os.sep + system_name ),
            {
                'use_shell':    True,
                'fizzle_bad_rc': False
            } )

        recover_packmol_ft = RecoverPackmolTask( {
            'dest':      self.output_prefix + os.sep + system_name,
            'recover':   True,
            'glob_patterns': [
                '*_restart.pack',
                '*_packmol.pdb',
                '*_packmol.pdb_FORCED',
                '*_packmol.inp'
            ],
            'forward_glob_patterns': {
                "packmol_pdb" : ["*_packmol.pdb_FORCED", "*_packmol.pdb"]
            }
        } )

        recover_packmol_fw = Firework( [ mkdir_ft, recover_packmol_ft ],
            spec={
                '_category':              self.std_worker,
                '_dupefinder':            DupeFinderExact(),
                '_allow_fizzled_parents': True,
                '_files_in':  {
                    'packmol_pdb' :       system_name + self._packmol_suffix + '.pdb',
                    'packmol_pdb_FORCED': system_name + self._packmol_suffix + '.pdb_FORCED'
                },
                '_files_out': {
                    'packmol_pdb' : system_name + self._packmol_suffix + '.pdb'
                },
                'system_name' :           system_name,
                'step':                   'recover_packmol',
                'geninfo':                self.geninfo()
            },
            name="{:s}_recover_packmol".format(system_name) )

        consecutive_fw_list.append(recover_packmol_fw)

    # pdb to gro
        packmol2gmx_ft =  ScriptTask.from_str(
            self._template_packmol2gmx_cmd.format(
                template_prefix = self.template_prefix,
                infile = system_name + self._packmol_suffix + '.pdb' ),
            {
                'stdout_file':  system_name + self._packmol2gmx_suffix + '.out',
                'stderr_file':  system_name + self._packmol2gmx_suffix + '.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        packmol2gmx_fw = Firework(
            packmol2gmx_ft,
            spec = {
                "_category":   self.std_worker,
                #"_allow_fizzled_parents": True,
                "_dupefinder": DupeFinderExact(),
                "_files_in":   {
                    "packmol_pdb" : system_name + self._packmol_suffix + '.pdb'
                    },
                "_files_out":  { "pdb_for_gmx":  system_name + '.pdb' },
                "system_name" : system_name,
                "step"   :     "packmol2gmx",
                'geninfo':     self.geninfo()
                },
            name=(system_name + self._packmol2gmx_suffix) ,
            parents=[recover_packmol_fw]
        )
        consecutive_fw_list.append(packmol2gmx_fw)

    # solvate in gromacs:
        gmx_solvate_fill_script_template_ft = TemplateWriterTask( {
            'context': {
                'header':     self.geninfo(),
                'system_name':system_name,
                'surfactant': surfactant,
                'ncation':    ncation,
                'nanion':     nanion,
                'box':        box_nanometer,
                'ionize':     False
            },
            'template_file': self.template_prefix + os.sep + 'gmx_solvate.sh',
            'output_file':   system_name + self._gmx_solvate_suffix + '.sh'} )

        single_surfactant_pdb = '1_{:s}.pdb'.format( surfactant )

        gmx_solvate_get_files_ft = FileTransferTask( {
                'files': [ {
                    'src':  self.template_prefix + os.sep + single_surfactant_pdb,
                    'dest': single_surfactant_pdb } ],
                'mode': 'copy' } )

        gmx_solvate_ft =  ScriptTask.from_str(
            self._template_gmx2pdb_cmd.format(
                infile = system_name + self._gmx_solvate_suffix + '.sh' ),
            {
                'stdout_file':  system_name + self._gmx_solvate_suffix + '.out',
                'stderr_file':  system_name + self._gmx_solvate_suffix + '.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            } )
        gmx_solvate_fw = Firework(
            [
                gmx_solvate_fill_script_template_ft,
                gmx_solvate_get_files_ft,
                gmx_solvate_ft
            ],
            spec={
                "_category":  self.queue_worker_serial,
                "_dupefinder": DupeFinderExact(),
                "_files_in":   { "pdb_for_gmx":  (system_name + '.pdb') },
                "_files_out":  {
                    "ionized_gro" : "{:s}_solvated.gro".format(system_name)
                    },
                "system_name": system_name,
                "step"   :     "gmx_solvate",
                'geninfo':     self.geninfo()
            },
            name="{:s}_gmx_solvate".format(system_name),
            parents=[packmol2gmx_fw])

        consecutive_fw_list.append(gmx_solvate_fw)

        # the system has already been ionized in packmol
        # thus, this step is skipped in GROMACS
        #copy_outfile_ft = FileTransferTask( {
        #    'files': [ {
        #        'src':  absolute_dir + os.sep + system_name + '_solvated.gro',
        #        'dest': absolute_dir + os.sep + system_name + '_ionized.gro' } ],
        #    'mode': 'copy' } )
        #copy_outfile_fw = Firework(copy_outfile_ft,
        #    spec={
        #        "_dupefinder": DupeFinderExact(),
        #        "system_name" :     system_name,
        #        "step"   :     "gmx_solvate_copy_outfile"
        #    },
        #    name="{:s}_copy_outfile".format(system_name))

    # convert to pdb chunks again
        pdb_segment_chunk_glob_pattern = '*_[0-9][0-9][0-9].pdb'

        gmx2pdb_fill_script_template_ft = TemplateWriterTask( {
            'context': {
                'header':       self.geninfo(),
                'system_name':  system_name,
            },
            'template_file': self.template_prefix + os.sep + 'gmx2pdb.sh',
            'output_file': system_name + self._gmx2pdb_suffix + '.sh'} )

        gmx2pdb_ft =  ScriptTask.from_str(
            self._template_gmx2pdb_cmd.format(
                infile = system_name + self._gmx2pdb_suffix + '.sh' ),
            {
                'stdout_file':  system_name + self._gmx2pdb_suffix + '.out',
                'stderr_file':  system_name + self._gmx2pdb_suffix + '.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        gmx2pdb_tar_ft = ScriptTask.from_str(
            'tar -czf {:s} {:s}'.format(
                system_name + '_segments.tar.gz',
                pdb_segment_chunk_glob_pattern ),
            {
                'stdout_file':  system_name + self._gmx2pdb_suffix + '_tar.out',
                'stderr_file':  system_name + self._gmx2pdb_suffix + '_tar.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        mkdir_ft =  ScriptTask.from_str(
            'mkdir -p "{:s}"'.format(
                self.output_prefix + os.sep + system_name ),
                {
                    'use_shell':    True, 'fizzle_bad_rc': False
                } )

        copy_output_files_ft = FileTransferTask( {
            'files': [
                system_name + '_ionized.gro',
                system_name + '_segments.tar.gz',
            ],
            'dest': self.output_prefix + os.sep + system_name + os.sep,
            'mode': 'copy' } )

        gmx2pdb_fw = Firework(
            [
                gmx2pdb_fill_script_template_ft,
                gmx2pdb_ft,
                gmx2pdb_tar_ft,
                mkdir_ft,
                copy_output_files_ft,

            ],
            spec={
                "_category":   self.std_worker,
                "_dupefinder": DupeFinderExact(),
                "_files_in":   {
                    "ionized_gro" : "{:s}_ionized.gro".format(system_name)
                },
                "_files_out":  {
                    "segments_tar":  "{:s}_segments.tar.gz".format(
                        system_name )
                },
                "system_name" :system_name,
                "step"   :     "gmx2pdb",
                'geninfo':     self.geninfo()
            },
            name="{:s}_gmx2pdb".format(system_name),
            parents=[gmx_solvate_fw])

        consecutive_fw_list.append(gmx2pdb_fw)

    ### PSFGEN

    # make psfgen input
        psfgen_untar_tf = ScriptTask.from_str('tar -xf {:s}'.format(
            system_name + '_segments.tar.gz'),
            {
                'stdout_file':  system_name + self._psfgen_suffix + '_untar.out',
                'stderr_file':  system_name + self._psfgen_suffix + '_untar.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        makeIdPdbDict_ft = MakeSegIdSegPdbDictTask( {
            'glob_pattern': pdb_segment_chunk_glob_pattern } )

        # in order to make FWAction be able to alter context content,
        # the following TemplateWriterTask uses global specs from the
        # encapsulating Firework
        psfgen_fill_script_template_ft = TemplateWriterTask( {
            'use_global_spec' : True} )
        #psfgen_fill_script_template_fw = Firework( psfgen_fill_script_template_ft,
        #    spec={
        #        "_dupefinder": DupeFinderExact(),
        #        "system_name"     : system_name,
        #        "step"       : "psfgen_fill_script_template",
        #    },
        #    name="{:s}_psfgen_fill_script_template".format(system_name))

    # get necessary files
        psfgen_get_files_ft = FileTransferTask( {
                'files': [ {
                    'src':  self.template_prefix \
                        + os.sep + 'par_all36_lipid_extended_stripped.prm',
                    'dest': 'par_all36_lipid_extended_stripped.prm' }, {
                    'src':  self.template_prefix + os.sep \
                        + 'top_all36_lipid_extended_stripped.rtf',
                    'dest': 'top_all36_lipid_extended_stripped.rtf' } ],
                'mode': 'copy' } )

    # run psfgen
        psfgen_ft =  ScriptTask.from_str(
            self._template_psfgen_cmd.format( infile = (
                system_name + self._psfgen_suffix + '.pgn')),
            {
                'stdout_file':  system_name + self._psfgen_suffix + '.out',
                'stderr_file':  system_name + self._psfgen_suffix + '.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        psfgen_fw = Firework(
            [
                psfgen_untar_tf,
                makeIdPdbDict_ft,
                psfgen_fill_script_template_ft,
                psfgen_get_files_ft,
                psfgen_ft
            ],
            spec={
                "_category":  self.std_worker,
                "_dupefinder": DupeFinderExact(),
                "_files_in":  {
                    "segments_tar": "{:s}_segments.tar.gz".format(
                        system_name ) },
                "_files_out": {
                    "psfgen_pdb": "{:s}_psfgen.pdb".format(system_name),
                    "psfgen_psf":    "{:s}_psfgen.psf".format(system_name)
                },
                "system_name": system_name,
                "step"       : "psfgen",
                # for the template writer task
                'context': {
                    'header':      self.geninfo(),
                    'system_name': system_name
                },
                'template_file': self.template_prefix + os.sep + 'psfgen.pgn',
                'output_file':   system_name + self._psfgen_suffix + '.pgn',
                'geninfo':       self.geninfo()
            },
            name = system_name +self._psfgen_suffix,
            parents=[gmx2pdb_fw])

        consecutive_fw_list.append(psfgen_fw)

    # run charmm2lammps
        ch2lmp_get_files_ft = FileTransferTask( {
                'files': [ {
                    'src':  self.template_prefix + os.sep \
                        + 'par_all36_lipid_extended_stripped.prm',
                    'dest': 'par_all36_lipid_extended_stripped.prm' }, {
                    'src':  self.template_prefix + os.sep \
                        + 'top_all36_lipid_extended_stripped.rtf',
                    'dest': 'top_all36_lipid_extended_stripped.rtf' } ],
                'mode': 'copy' } )

        ch2lmp_ft =  ScriptTask.from_str(
                self._template_ch2lmp_cmd.format(
                    system_name = system_name, box = box_angstrom), {
                'stdout_file':  system_name + self._ch2lmp_suffix + '.out',
                'stderr_file':  system_name + self._ch2lmp_suffix + '.err',
                'use_shell':    True,
                'fizzle_bad_rc':True } )

        # If existent, just deletes existing files
        file_identifier = "{:s}_psfgen.data".format(system_name)

        ch2lmp_delete_ft = DeleteFilesTask(
            {
                'identifiers': [file_identifier]
            }
        )

        ch2lmp_store_ft =  AddFilesTask(
            {
                'paths':       ["{:s}_psfgen.data".format(system_name)],
                'identifiers': [ file_identifier ],
                'metadata': {
                    'system_name':          system_name,
                    'sb_name':              sb_name,
                    'surfactant':           row["surfactant"],
                    'substrate':            row["substrate"],
                    'counterion':           row["counterion"],
                    'solvent':              row["solvent"],
                    'sf_preassembly':       row["sf_preassembly"],
                    'ci_preassembly':       row["ci_initial_placement"],
                    'sb_crystal_plane':     row["sb_crystal_plane"]
                }
            }
        )

        copy_output_files_ft = FileTransferTask( {
            'files': [
                "{:s}_psfgen.data".format(system_name),
                "{:s}_psfgen.in".format(system_name),
                "{:s}_psfgen_ctrl.pdb".format(system_name),
                "{:s}_psfgen_ctrl.psf".format(system_name)
            ],
            'dest': self.output_prefix + os.sep + system_name,
            'mode': 'copy' } )

        ch2lmp_fw = Firework(
            [
                ch2lmp_get_files_ft,
                ch2lmp_ft,
                ch2lmp_delete_ft,
                ch2lmp_store_ft,
                mkdir_ft,
                copy_output_files_ft
            ],
            spec={
                "_category":   self.std_worker,
                "_dupefinder": DupeFinderExact(),
                "_files_in": {
                    "psfgen_pdb":    "{:s}_psfgen.pdb".format(system_name),
                    "psfgen_psf":    "{:s}_psfgen.psf".format(system_name) },
                "_files_out": {
                    "ch2lmp_data":      "{:s}_psfgen.data".format(system_name),
                    "ch2lmp_in":        "{:s}_psfgen.in".format(system_name),
                    "ch2lmp_ctrl_pdb":  "{:s}_psfgen_ctrl.pdb".format(system_name),
                    "ch2lmp_ctrl_psf":  "{:s}_psfgen_ctrl.psf".format(system_name)
                },
                "system_name": system_name,
                "step"       : "ch2lmp",
                'geninfo':     self.geninfo()
            },
            name= system_name + self._ch2lmp_suffix,
            parents=[psfgen_fw])

        consecutive_fw_list.append(ch2lmp_fw)

        # workflow
        parent_links = { consecutive_fw_list[i] : consecutive_fw_list[i+1] \
                        for i in range(len(consecutive_fw_list)-1) }

        return Workflow( consecutive_fw_list, parent_links,
            name="{:s}_prep_wf".format(system_name) )


# MD in LAMMPS (originally on JUWELS, to be tested)
    def prepare_systems(  self, system_name ):
        """Initiates workflows for all systems in system_names"""

        # prepare_system_files_fw_list = []
        # for system_name in system_names:
        # A LAMMPS data file file suffixed _psfgen.data
        # is expected to exist in the FilePad
        data_file = '{:s}_psfgen.data'.format(system_name)

        # standard LAMMPS input files lmp_header.input and
        # lmp_minimization.input are expected to reside within
        # template _prefix
        #get_input_files_ft = FileTransferTask( {
        #    'files': [
        #        {
        #            'src':  self.template_prefix + os.sep + 'lmp_header.input',
        #            'dest': 'lmp_header.input'
        #        }, {
        #            'src':  self.template_prefix + os.sep + 'lmp_minimization.input',
        #            'dest': 'lmp_minimization.input'
        #        }
        #    ],
        #    'mode': 'copy' } )

        get_data_file_ft = GetFilesTask( {
                'identifiers': [ data_file ],
                'new_file_names': [ data_file ] } )

        # write a tiny one-line file to tell lammps the system name
        # might be changed to -v baseName at LAMMPS command line instead
        write_system_specific_input_ft = FileWriteTask( {
            'files_to_write': [
                {
                    'filename': 'system_specific.input',
                    'contents':  'variable baseName string {:s}'.format(
                        system_name )
                }
            ] } )

        # creates output directory if necessary
        # better replace by in-python task at some point
        mkdir_ft =  ScriptTask.from_str(
            'mkdir -p "{:s}"'.format(self.output_prefix + os.sep + system_name),
                {
                    'use_shell':    True, 'fizzle_bad_rc': False
                } )

        copy_output_files_ft = FileTransferTask( {
            'files': [
                data_file
            ],
            'dest': self.output_prefix + os.sep + system_name + os.sep,
            'mode': 'copy' } )

        prepare_system_files_fw = Firework(
            [
                #get_input_files_ft,
                get_data_file_ft,
                write_system_specific_input_ft,
                mkdir_ft,
                copy_output_files_ft
            ],
            spec={
                "_category":   self.std_worker,
                "_dupefinder": DupeFinderExact(),
                "_files_out":  {
                    "data_file"   :         data_file,
                    "system_specific_file": 'system_specific.input'
                },
                "system_name": system_name,
                "step"     :   "prepare_system_files",
                "geninfo":     self.geninfo() # serves to distinguish duplicates
            },
            name="{:s}_prepare_system_files".format(system_name))

        return prepare_system_files_fw
        # prepare_system_files_fw_list.append((system_name,prepare_system_files_fw))

        #prepare_system_files_fw_dict = dict(prepare_system_files_fw_list)
        # returns a system_name : Firework dict
        #return prepare_system_files_fw_dict

    def prepare_minimizations( self, system_names, robust = False ):
        """Prepares LAMMPS minimizations.
        Paramters
        ---------
        system_names: list of str
            Subset of systems to process.
        robust: bool
            Passes additional command line switch
            '-v robust_minimization 1' to LAMMPS executable,
            causing lmp_minimization.input issues a
            "neigh_modify delay 0 every 1 check yes one 5000 page 250000"
            statement. See LAMMPS documentation for more info.
        """

        lmp_cmd_suffix = ' -v robust_minimization 1' if robust else  ''

        minimization_fw_list = []
        for system_name in system_names:
            get_input_files_ft = FileTransferTask( {
                'files': [
                    {
                        'src':  self.template_prefix + os.sep + 'lmp_header.input',
                        'dest': 'lmp_header.input'
                    }, {
                        'src':  self.template_prefix + os.sep + 'lmp_minimization.input',
                        'dest': 'lmp_minimization.input'
                    }
                ],
                'mode': 'copy' } )

            minimization_ft =  ScriptTask.from_str(
                self.lmp_cmd.format(
                    inputFile = 'lmp_minimization.input' ) + lmp_cmd_suffix,
                {
                    'stdout_file':  system_name + '_minimization.out',
                    'stderr_file':  system_name + '_minimization.err',
                    'use_shell':    True,
                    'fizzle_bad_rc':True
                })

            extract_thermo_ft =  ScriptTask.from_str(
                self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
                    system_name + '_minimization.log',
                    system_name + '_minimization_thermo.out'),
                    {
                        'use_shell':    True, 'fizzle_bad_rc': False
                    } )

            mkdir_ft =  ScriptTask.from_str(
                'mkdir -p "{:s}"'.format(self.output_prefix + os.sep + system_name),
                    {
                        'use_shell':    True, 'fizzle_bad_rc': False
                    } )

            copy_output_files_ft = FileTransferTask( {
                'files': [
                    system_name + '_minimized.lammps',
                    system_name + '_minimization.log',
                    system_name + '_minimization_thermo.out'
                ],
                'dest': self.output_prefix + os.sep + system_name,
                'mode': 'copy' } )

            minimization_fw = Firework(
                [
                    get_input_files_ft,
                    minimization_ft,
                    extract_thermo_ft,
                    mkdir_ft,
                    copy_output_files_ft
                ],
                spec={
                    "_queueadapter": self.minimization_queueadapter,
                    "_category":     self.queue_worker,
                    "_dupefinder":   DupeFinderExact(),
                    "_files_in":  {
                        "data_file"   :         system_name + '_psfgen.data',
                        "system_specific_file": 'system_specific.input'
                    },
                    "_files_out":  {
                        "data_file"   :         system_name + '_minimized.lammps',
                        "system_specific_file": 'system_specific.input'
                    },
                    "system_name": system_name,
                    "step"     :   "minimization",
                    "geninfo":     self.geninfo() # serves to distinguish duplicates
                },
                name="{:s}_minimzation".format(system_name))

            minimization_fw_list.append((system_name,minimization_fw))

        system_name_minimization_fw_dict = dict(minimization_fw_list)
        return system_name_minimization_fw_dict

    def prepare_nvt_equlibrations( self, system_names ):
        equilibration_nvt_fw_list = []
        for system_name in system_names:
            get_input_files_ft = FileTransferTask( {
                'files': [
                    {
                        'src':  self.template_prefix + os.sep + 'lmp_header.input',
                        'dest': 'lmp_header.input'
                    }, {
                        'src':  self.template_prefix + os.sep + 'lmp_equilibration_nvt.input',
                        'dest': 'lmp_equilibration_nvt.input'
                    }
                ],
                'mode': 'copy' } )

            # for now, switch mpiio off with mechanism in lammps input script
            equilibration_nvt_ft =  ScriptTask.from_str(
                self.lmp_cmd.format(inputFile = 'lmp_equilibration_nvt.input'),
                {
                    'stdout_file':  system_name + '_equilibration_nvt.out',
                    'stderr_file':  system_name + '_equilibration_nvt.err',
                    'use_shell':    True,
                    'fizzle_bad_rc':True
                })

            extract_thermo_ft =  ScriptTask.from_str(
                self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
                    system_name + '_nvtEquilibration.log',
                    system_name + '_nvtEquilibration_thermo.out'),
                    {
                        'use_shell':    True, 'fizzle_bad_rc': False
                    } )

            mkdir_ft =  ScriptTask.from_str(
                'mkdir -p "{:s}"'.format(self.output_prefix + os.sep + system_name),
                    {
                        'use_shell':    True, 'fizzle_bad_rc': False
                    } )

            copy_output_files_ft = FileTransferTask( {
                'files': [
                    system_name + '_nvtEquilibrated.lammps',
                    system_name + '_nvtEquilibration.log',
                    system_name + '_nvtEquilibration_thermo.out'
                ],
                'dest': self.output_prefix + os.sep + system_name + os.sep,
                'mode': 'copy' } )

            equilibration_nvt_fw = Firework(
                [
                    get_input_files_ft,
                    equilibration_nvt_ft,
                    extract_thermo_ft,
                    mkdir_ft,
                    copy_output_files_ft
                ],
                spec={
                    "_queueadapter": self.equilibration_queueadapter,
                    "_category":   self.queue_worker,
                    "_dupefinder": DupeFinderExact(),
                    "_files_in":  {
                        "data_file"   :         system_name + '_minimized.lammps',
                        "system_specific_file": 'system_specific.input'
                    },
                    "_files_out":  {
                        "data_file"   :         system_name + '_nvtEquilibrated.lammps',
                        "system_specific_file": 'system_specific.input'
                    },
                    "system_name": system_name,
                    "step"     :   "equilibration_nvt",
                    "geninfo":     self.geninfo() # serves to distinguish duplicates
                },
                name="{:s}_equilibration_nvt".format(system_name))

            equilibration_nvt_fw_list.append((system_name,equilibration_nvt_fw))

        system_name_equilibration_nvt_fw_dict = dict(equilibration_nvt_fw_list)
        return system_name_equilibration_nvt_fw_dict

    def prepare_npt_equlibrations( self, system_names ):
        equilibration_npt_fw_list = []
        for system_name in system_names:
            get_input_files_ft = FileTransferTask( {
                'files': [
                    {
                        'src':  self.template_prefix + os.sep + 'lmp_header.input',
                        'dest': 'lmp_header.input'
                    }, {
                        'src':  self.template_prefix + os.sep + 'lmp_equilibration_npt.input',
                        'dest': 'lmp_equilibration_npt.input'
                    }
                ],
                'mode': 'copy' } )

            # for now, switxch mpiio off with mechanism in lammps input script
            equilibration_npt_ft =  ScriptTask.from_str(
                self.lmp_cmd.format(inputFile='lmp_equilibration_npt.input'),
                {
                    'stdout_file':  system_name + '_equilibration_npt.out',
                    'stderr_file':  system_name + '_equilibration_npt.err',
                    'use_shell':    True,
                    'fizzle_bad_rc':True
                })

            extract_thermo_ft =  ScriptTask.from_str(
                self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
                    system_name + '_nptEquilibration.log',
                    system_name + '_nptEquilibration_thermo.out'),
                    {
                        'use_shell':    True, 'fizzle_bad_rc': False
                    } )

            mkdir_ft =  ScriptTask.from_str(
                'mkdir -p "{:s}"'.format(self.output_prefix + os.sep + system_name),
                    {
                        'use_shell':    True, 'fizzle_bad_rc': False
                    } )

            copy_output_files_ft = FileTransferTask( {
                'files': [
                    system_name + '_nptEquilibrated.lammps',
                    system_name + '_nptEquilibration.log',
                    system_name + '_nptEquilibration_thermo.out'
                ],
                'dest': self.output_prefix + os.sep + system_name + os.sep,
                'mode': 'copy' } )

            equilibration_npt_fw = Firework(
                [
                    get_input_files_ft,
                    equilibration_npt_ft,
                    extract_thermo_ft,
                    mkdir_ft,
                    copy_output_files_ft
                ],
                spec={
                    "_queueadapter": self.equilibration_queueadapter,
                    "_category":   self.queue_worker,
                    "_dupefinder": DupeFinderExact(),
                    "_files_in":  {
                        "data_file"   :         system_name + '_nvtEquilibrated.lammps',
                        "system_specific_file": 'system_specific.input'
                    },
                    "_files_out":  {
                        "data_file"   :         system_name + '_nptEquilibrated.lammps',
                        "system_specific_file": 'system_specific.input'
                    },
                    "system_name": system_name,
                    "step"     :   "equilibration_npt",
                    "geninfo":     self.geninfo() # serves to distinguish duplicates
                },
                name="{:s}_equilibration_npt".format(system_name))

            equilibration_npt_fw_list.append((system_name,equilibration_npt_fw))

        system_name_equilibration_npt_fw_dict = dict(equilibration_npt_fw_list)
        return system_name_equilibration_npt_fw_dict

    #
    def prepare_production( self, system_names ):
        production_fw_list = []
        for system_name in system_names:
            get_input_files_ft = FileTransferTask( {
                'files': [
                    {
                        'src':  self.template_prefix + os.sep + 'lmp_header.input',
                        'dest': 'lmp_header.input'
                    }, {
                        'src':  self.template_prefix + os.sep + 'lmp_production_mixed.input',
                        'dest': 'lmp_production_mixed.input'
                    }, {
                        'src':  self.template_prefix + os.sep + 'lmp_10ns_production_mixed.input',
                        'dest': 'lmp_10ns_production_mixed.input'
                    }
                ],
                'mode': 'copy' } )


            lmp_cmd = ' '.join((
                self.lmp_cmd.format(inputFile='lmp_10ns_production_mixed.input'),
                '-v baseName {baseName:s}'.format(baseName=system_name) ))

            production_ft =  ScriptTask.from_str(
                lmp_cmd,
                {
                    'stdout_file':  system_name + '_10ns_production.out',
                    'stderr_file':  system_name + '_10ns_production.err',
                    'use_shell':    True,
                    'fizzle_bad_rc':True
                })

            extract_thermo_ft =  ScriptTask.from_str(
                self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
                    system_name + '_10ns_production_mixed.log',
                    system_name + '_10ns_production_mixed_thermo.out'),
                    {
                        'use_shell':    True, 'fizzle_bad_rc': False
                    } )

            mkdir_ft =  ScriptTask.from_str(
                'mkdir -p "{:s}"'.format(self.output_prefix + os.sep + system_name),
                    {
                        'use_shell':    True, 'fizzle_bad_rc': False
                    } )

            copy_output_files_ft = FileTransferTask( {
                'files': [
                    system_name + '_production_mixed.lammps',
                    system_name + '_10ns_production_mixed.log',
                    system_name + '_10ns_production_mixed_thermo.out'
                ],
                'dest': self.output_prefix + os.sep + system_name + os.sep,
                'mode': 'copy' } )

            production_fw = Firework(
                [
                    get_input_files_ft,
                    production_ft,
                    extract_thermo_ft,
                    mkdir_ft,
                    copy_output_files_ft
                ],
                spec={
                    "_queueadapter": self.production_queueadapter,
                    "_category":   self.queue_worker,
                    "_dupefinder": DupeFinderExact(),
                    "_files_in":  {
                        "data_file"   :         system_name + '_nptEquilibrated.lammps',
                    },
                    "_files_out":  {
                        "data_file"   :         system_name + '_production_mixed.lammps',
                    },
                    "system_name": system_name,
                    "step"     :   "10ns_production_mixed",
                    "geninfo":     self.geninfo() # serves to distinguish duplicates
                },
                name="{:s}_10ns_production_mixed".format(system_name))

            production_fw_list.append((system_name,production_fw))

        system_name_production_fw_dict = dict(production_fw_list)
        return system_name_production_fw_dict

    # tuned to continue runs by self.colvars_production or self.production
    def prepare_restart( self, system_name,
        lmp_suffix_template = ' '.join(('-v baseName {baseName:s}',
            '-v has_indenter 1 -v pbc2d 0 -v mpiio 0 -v use_colvars 1')),
        output_subfolder = 'production' ):
        #restart_fw_list = []
        #for system_name in system_names:
        #lmp_cmd = ' '.join((self._template_lmp_cmd.format(
        #    inputFile = 'lmp_production.input' ), lmp_suffix_template.format(
        #        baseName=system_name, dataFile='datafile.lammps' ) ))

        recover_lammps_ft = RecoverLammpsTask( {
            'restart_file_glob_pattern': ['*.restart[0-9]'],
            'fizzle_on_no_restart_file': True, # only if restart neccessary
            'dest': self.output_prefix + os.sep + system_name + \
                os.sep + output_subfolder,
            'other_glob_patterns': [
                '*.nc', # netcdf trajectory
                '*.log', # LAMMPS logfile
                '*.colvars.traj', # colvars outputs
                '*.colvars.state',
                '*.ti.count',
                '*.ti.grad',
                '*.ti.pmf' ],
            'file_transfer_mode': 'copy',
            'continue': True,
            'recover':  True,
            'lmp_queueadapter': self.production_queueadapter,
            'lmp_cmd': self._template_lmp_cmd.format(
                inputFile='lmp_production.input'),
            'lmp_opt': lmp_suffix_template
        } )

        recover_lammps_fw = Firework( recover_lammps_ft,
            spec = {
                '_category':                   self.std_worker,
                '_allow_fizzled_parents':      True,
                'system_name':                 system_name,
                'step':                        'recover_lammps',
                "geninfo":     self.geninfo() # serves to distinguish duplicates
                },
            name    = '{:s}_recover_lammps'.format(system_name) )

        #restart_fw_list.append((system_name,recover_lammps_fw))

        #system_name_restart_fw_dict = dict(restart_fw_list)
        #return system_name_restart_fw_dict
        return recover_lammps_fw

    def extract_frame( self, system_name,
        topology_file_name = None, trajectory_file_name = None,
        frame = 0, netcdf2data_cmd_suffix = '', additional_spec = {} ):

        # if no topology file specified, pull from previous step
        if topology_file_name is None:
            topology_file_name = 'datafile.lammps'
            additional_spec["_files_in"] = {"data_file": 'datafile.lammps'}

        netcdf2data_cmd_suffix += '--frames {frame:d}'

        # assumes output pattern 'frame_{:d}.lammps'
        netcdf2data_cmd = ' '.join((
            self._template_netcdf2data_cmd.format(
                datafile = topology_file_name, trajfile = trajectory_file_name),
            netcdf2data_cmd_suffix.format( frame=int(frame) ) ))

        netcdf2data_ft =  ScriptTask.from_str(
            netcdf2data_cmd,
            {
                'stdout_file':  'netcdf2data.out',
                'stderr_file':  'netcdf2data.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        # TODO: correct fixed file name
        fw_spec = {
            "_category":   self.std_worker,
            "_dupefinder": DupeFinderExact(),
            "_files_out": {
                "data_file": "frame_{:08d}.lammps".format(frame)
            },
            "system_name": system_name,
            "step"     :   "netcdf2frame",
            "geninfo":     self.geninfo() # serves to distinguish duplicates
        }

        if len(additional_spec) > 0:
            fw_spec.update( additional_spec )

        netcdf2data_fw = Firework(
            [ netcdf2data_ft ], spec = fw_spec,
            name= system_name + '_netcdf2frame_{:d}'.format(frame) )

        return netcdf2data_fw


    def store_data_file(
        self, system_name, file_identifier = None, file_name = None,
        additional_spec = {} ):

        if file_identifier is None:
            file_identifier = "{:s}_10ns.lammps".format(system_name)
        if file_name is None:
            file_name = file_identifier

        # if existent, just deletes existing file
        delete_ft = DeleteFilesTask(
            {
                'identifiers': [file_identifier]
            }
        )

        store_ft =  AddFilesTask(
            {
                'paths':       [ file_name ],
                'identifiers': [ file_identifier ],
                'metadata': {
                    'system_name':          system_name,
                    # 'sb_name':              row["sb_name"],
                    # 'surfactant':           row["surfactant"],
                    # 'substrate':            row["substrate"],
                    # 'counterion':           row["counterion"],
                    # 'solvent':              row["solvent"],
                    # 'sf_preassembly':       row["sf_preassembly"],
                    # 'ci_preassembly':       row["ci_initial_placement"],
                    # 'sb_crystal_plane':     row["sb_crystal_plane"],
                }
            }
        )

        # TODO: correct fixed file name
        fw_spec = {
            "_category":     self.std_worker,
            "_dupefinder":   DupeFinderExact(),
            "_files_in": {
                "data_file": file_name
            },
            "system_name": system_name,
            "step"     :   "store_data_file",
            "geninfo":     self.geninfo() # serves to distinguish duplicates
        }

        if len(additional_spec) > 0:
            fw_spec.update( additional_spec )


        fw = Firework(
            [
                delete_ft,
                store_ft
            ],
            spec = fw_spec,
            name="{:s}_store_data_file".format(system_name) )
        return fw

    def pull_datafile_from_db(
        self, system_name, data_file_identifier = None, data_file_identifier_suffix = '_initial.lammps', fworker = None ):

        if data_file_identifier is None:
            data_file_identifier = system_name + data_file_identifier_suffix

        if fworker is None: fworker = self.std_worker

        # read from data base
        get_data_files_ft = GetFilesTask( {
                'identifiers': [ data_file_identifier ],
                'new_file_names': [ 'datafile.lammps' ] } )

        fw = Firework(
            [ get_data_files_ft ],
            spec={
                "_category":     fworker,
                #"_dupefinder":   DupeFinderExact(),
                "_files_out":  {
                    "data_file": 'datafile.lammps'
                },
                "system_name": system_name,
                "step"     :   "pull_datafile_from_db",
                "geninfo":     self.geninfo() # serves to distinguish duplicates
            },
            name="{:s}_pull_datafile_from_db".format(system_name))
        return fw

    def initiate_indenter_workflow(
        self, system_name, interface_file, indenter_file):

        # read from data base
        get_data_files_ft = GetFilesTask( {
                'identifiers': [ indenter_file, interface_file ],
                'new_file_names': [ indenter_file, interface_file ] } )

        fw = Firework(
            [ get_data_files_ft ],
            spec={
                "_category":     self.std_worker,
                "_dupefinder":   DupeFinderExact(),
                "_files_out":  {
                    "interface_file": interface_file,
                    "indenter_file":  indenter_file
                },
                "system_name": system_name,
                "step"     :   "initiate_indenter_workflow",
                "geninfo":     self.geninfo() # serves to distinguish duplicates
            },
            name="{:s}_initiate_indenter_workflow".format(system_name))

        return fw

    def indenter_insertion(
        self, system_name, fworker = None, desired_distance = 100.0 ):
        """Insert indenter.
        Paramters
        ---------
        system_name: str
        """
        if fworker is None: fworker = self.std_worker

        row = self._sim_df.loc[system_name,:]
        surfactant  = row["surfactant"]

        interface_file = 'interface.lammps'
        indenter_file =  'indenter.pdb'

        get_input_files_ft = GetFilesTask( {
             'identifiers': [ 'jlh_vmd.tcl', 'indenter_insertion.tcl' ],
             'new_file_names': [ 'jlh_vmd.tcl', 'indenter_insertion.template' ] } )

        #get_input_files_ft = FileTransferTask( {
        #    'files': [
        #        {
        #            'src':  self.template_prefix + os.sep + 'jlh_vmd.tcl',
        #            'dest': 'jlh_vmd.tcl'
        #        }
        #    ],
        #    'mode': 'copy' } )

        indenter_insertion_fill_script_template_ft = TemplateWriterTask( {
            'context' :      {
                'header':           self.geninfo(),
                'surfactant':       surfactant,
                'desired_distance': desired_distance,
                'interface_file':   interface_file,
                'indenter_file':    indenter_file,
                'output_prefix':    system_name
            },
            'template_file': 'indenter_insertion.template',
            'output_file':   'indenter_insertion.tcl' } )

        indenter_insertion_ft =  ScriptTask.from_str(
            self._template_vmd_cmd.format(
                infile = 'indenter_insertion.tcl' ),
            {
                'stdout_file':  'indenter_insertion.out',
                'stderr_file':  'indenter_insertion.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        convert_ft =  ScriptTask.from_str(
            self._template_convert_cmd.format(
                infile  = system_name + '.tga',
                outfile = system_name + '.png'),
                {
                    'use_shell':     True,
                    'fizzle_bad_rc': False
                } )

        #dest = self.output_prefix + os.sep + system_name + os.sep + 'indenter_immersed'
        #mkdir_ft =  ScriptTask.from_str(
        #    'mkdir -p "{:s}"'.format(dest),
        #        {
        #            'use_shell':     True,
        #            'fizzle_bad_rc': False
        #        } )

        #copy_output_files_ft = FileTransferTask( {
        #    'files': [
        #        system_name + '.lammps',
        #        system_name + '.psf',
        #        system_name + '.pdb',
        #        system_name + '.png',
        #        'indenter_insertion.out', # for checking vmd tcl log file
        #    ],
        #    'dest': dest,
        #    'mode': 'copy' } )

        fw = Firework(
            [
                get_input_files_ft,
                indenter_insertion_fill_script_template_ft,
                indenter_insertion_ft,
                convert_ft
                #mkdir_ft,
                #copy_output_files_ft
            ],
            spec={
                #"_queueadapter": self.minimization_queueadapter,
                "_category":     fworker,
                #"_dupefinder":   DupeFinderExact(),
                "_files_in":  {
                    "interface_file" :     interface_file,
                    "indenter_file"  :     indenter_file
                },
                "_files_out":  {
                    "pdb_file": system_name + '.pdb',
                    "psf_file": system_name + '.psf',
                    "data_file": system_name + '.lammps'
                },
                "system_name": system_name,
                "step"     :   "indenter_insertion",
                "geninfo":     self.geninfo() # serves to distinguish duplicates
            },
            name="{:s}_indenter_insertion".format(system_name))
        return fw

    def add_parameters_to_datafile_and_store(
        self, system_name, reffile_identifier = None, outfile_identifier = None,
        indenter_suffix = None, reffile_suffix = '_psfgen.data',
        outfile_suffix = '_initial.lammps',
        output_subfolder = 'datafiles_merged' ):

        previous_system_name = system_name
        if indenter_suffix is not None:
            if system_name.endswith(indenter_suffix) and len(indenter_suffix) > 0:
                previous_system_name = system_name[:-len(indenter_suffix)]

        if reffile_identifier is None:
            reffile_identifier = previous_system_name + reffile_suffix

        if outfile_identifier is None:
            outfile_identifier = system_name + outfile_suffix

        # pull reference file from data base
        get_data_files_ft = GetFilesTask( {
                'identifiers':    [ reffile_identifier ],
                'new_file_names': [ 'reffile.lammps' ] } )

        # MergeLammpsDataFiles contains functionality,
        # merge.py is pizza.py-callable wrapper
        # get_input_files_ft = FileTransferTask( {
        #    'files': [
        #        {
        #            'src':  self.template_prefix + os.sep + 'MergeLammpsDataFiles.py',
        #            'dest': 'MergeLammpsDataFiles.py'
        #        },
        #        {
        #            'src':  self.template_prefix + os.sep + 'merge.py',
        #            'dest': 'merge.py'
        #        }
        #    ],
        #    'mode': 'copy' } )
        get_input_files_ft = GetFilesTask( {
                'identifiers':    [ 'MergeLammpsDataFiles.py', 'merge.py' ],
                'new_file_names': [ 'MergeLammpsDataFiles.py', 'merge.py' ] } )

        pizzapy_merge_ft =  ScriptTask.from_str(
            self._template_pizzapy_merge_cmd.format(
                datafile = 'datafile.lammps',
                reffile  = 'reffile.lammps',
                outfile  = outfile_identifier ),
            {
                'stdout_file':  'pizzapy_merge.out',
                'stderr_file':  'pizzapy_merge.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        # if existent, just deletes existing file
        delete_ft = DeleteFilesTask(
            {
                'identifiers': [ outfile_identifier ]
            }
        )

        store_ft =  AddFilesTask(
            {
                'paths':       [ outfile_identifier ],
                'identifiers': [ outfile_identifier ],
                'metadata': {
                    'system_name':          system_name
                }
            }
        )

        # dest = self.output_prefix + os.sep + system_name + os.sep + output_subfolder
        # mkdir_ft =  ScriptTask.from_str(
        #     'mkdir -p "{:s}"'.format(dest),
        #        {
        #            'use_shell':     True,
        #            'fizzle_bad_rc': False
        #        } )

        # copy_output_files_ft = FileTransferTask( {
        #   'files': [
        #        outfile_identifier,
        #        'pizzapy_merge.out', # for checking vmd tcl log file
        #    ],
        #    'dest': dest,
        #    'mode': 'copy' } )

        fw = Firework(
            [
                get_data_files_ft,
                get_input_files_ft,
                pizzapy_merge_ft,
                delete_ft,
                store_ft,
                # mkdir_ft,
                # copy_output_files_ft
            ],
            spec={
                "_category":     self.std_worker,
                # "_dupefinder":   DupeFinderExact(),
                "_files_in": {
                    "data_file": 'datafile.lammps'
                },
                "_files_out": {
                    "data_file": outfile_identifier
                },
                "system_name": system_name,
                "step"     :   "pizzapy_merge",
                "geninfo":     self.geninfo() # serves to distinguish duplicates
            },
            name="{:s}_pizzapy_merge".format(system_name))
        return fw

    # LAMMPS

    def minimize(self, system_name,
        lmp_suffix_template='-v baseName {baseName:s} -v dataFile {dataFile:s}'):
        """
        Sample for lmp_suffix: for a call like
            srun lmp -in lmp_minimization.input \
                -v has_indenter 1 -v robust_minimization 0 -v pbc2d 1 \
                -v baseName 377_SDS_on_AU_111_51x30x2_monolayer \
                -v dataFile 377_SDS_on_AU_111_51x30x2_monolayer.lammps
        set lmp_suffix='-v has_indenter 1 -v robust_minimization 0 -v pbc2d 1 \
            -v baseName {baseName:s} -v dataFile {dataFile:s}'
        """

        get_input_files_ft = GetFilesTask( {
                'identifiers':    [ 'lmp_header.input', 'lmp_minimization.input', 'extract_thermo.sh' ] } )
        # get_input_files_ft = FileTransferTask( {
        #    'files': [
        #        {
        #            'src':  self.template_prefix + os.sep + 'lmp_header.input',
        #            'dest': 'lmp_header.input'
        #        }, {
        #            'src':  self.template_prefix + os.sep + 'lmp_minimization.input',
        #            'dest': 'lmp_minimization.input'
        #        }
        #    ],
        #    'mode': 'copy' } )

        lmp_cmd = ' '.join((self._template_lmp_cmd.format(
            inputFile = 'lmp_minimization.input' ), lmp_suffix_template.format(
                baseName=system_name, dataFile='datafile.lammps' ) ))

        minimization_ft =  ScriptTask.from_str(
            lmp_cmd,
            {
                'stdout_file':  'lmp_minimization.out',
                'stderr_file':  'lmp_minimization.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        extract_thermo_ft =  ScriptTask.from_str(
            'bash extract_thermo.sh {:s} {:s}'.format(
                system_name + '_minimization.log',
                system_name + '_minimization_thermo.out'),
                {
                    'use_shell':    True, 'fizzle_bad_rc': False
                } )

        # dest = self.output_prefix + os.sep + system_name + os.sep + 'minimized'
        # mkdir_ft =  ScriptTask.from_str(
        #    'mkdir -p "{:s}"'.format(dest),
        #        {
        #            'use_shell':     True,
        #            'fizzle_bad_rc': False
        #        } )

        # copy_output_files_ft = FileTransferTask( {
        #    'files': [
        #        system_name + '_minimized.lammps',
        #        system_name + '_minimization.log',
        #        system_name + '_minimization_thermo.out',
        #        'lmp_minimization.out',
        #    ],
        #    'dest': dest,
        #    'mode': 'copy' } )

        fw = Firework(
            [
                get_input_files_ft,
                minimization_ft,
                extract_thermo_ft,
                #mkdir_ft,
                #copy_output_files_ft
            ],
            spec={
                "_queueadapter": self.minimization_queueadapter,
                "_category":     self.queue_worker,
                #"_dupefinder":   DupeFinderExact(),
                "_files_in":  {
                    "data_file"   :  'datafile.lammps',
                },
                "_files_out":  {
                    "data_file"   :         system_name + '_minimized.lammps',
                },
                "system_name": system_name,
                "step"     :   "minimization",
                "geninfo":     self.geninfo() # serves to distinguish duplicates
            },
            name="{:s}_minimzation".format(system_name))
        return fw

    def nvtEquilibrate(self, system_name,
        lmp_suffix_template='-v baseName {baseName:s} -v dataFile {dataFile:s}'):
        """
        Sample for lmp_suffix: for a call like
            srun lmp -in lmp_equilibration_nvt.input \
                -v has_indenter 0 -v pbc2d 0 -v reinitialize_velocities 1\
                -v baseName 377_SDS_on_AU_111_51x30x2_monolayer \
                -v dataFile 377_SDS_on_AU_111_51x30x2_monolayer.lammps
        set lmp_suffix='-v has_indenter 0 -v reinitialize_velocities 1 \
            -v baseName {baseName:s} -v dataFile {dataFile:s}'
        """

        get_input_files_ft = GetFilesTask( {
                'identifiers':    [ 'lmp_header.input', 'lmp_equilibration_nvt.input', 'extract_thermo.sh' ] } )
        #get_input_files_ft = FileTransferTask( {
        #    'files': [
        #        {
        #            'src':  self.template_prefix + os.sep + 'lmp_header.input',
        #            'dest': 'lmp_header.input'
        #        }, {
        #            'src':  self.template_prefix + os.sep + 'lmp_equilibration_nvt.input',
        #            'dest': 'lmp_equilibration_nvt.input'
        #        }
        #    ],
        #    'mode': 'copy' } )

        lmp_cmd = ' '.join((self._template_lmp_cmd.format(
            inputFile = 'lmp_equilibration_nvt.input' ), lmp_suffix_template.format(
                baseName=system_name, dataFile='datafile.lammps' ) ) )

        lmp_ft =  ScriptTask.from_str(
            lmp_cmd,
            {
                'stdout_file':  'lmp_nvtEquilibration.out',
                'stderr_file':  'lmp_nvtEquilibration.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        extract_thermo_ft =  ScriptTask.from_str(
            'bash extract_thermo.sh {:s} {:s}'.format(
                system_name + '_nvtEquilibration.log',
                system_name + '_nvtEquilibration_thermo.out'),
                {
                    'use_shell':    True, 'fizzle_bad_rc': False
                } )

        #tail_ft_list = self.get_post_lammps_tasks( system_name,
        #    step = 'equilibration_nvt',
        #    suffix = '_nvtEquilibration',
        #    lmp_suffix = '_nvtEquilibrated',
        #    output_folder = 'nvtEquilibrated' )

        fw = Firework(
            [
                get_input_files_ft,
                lmp_ft,
		extract_thermo_ft
                #*tail_ft_list
            ],
            spec={
                "_queueadapter": self.equilibration_queueadapter,
                "_category":     self.queue_worker,
                #"_dupefinder":   DupeFinderExact(),
                "_files_in":  {
                    "data_file"   :  'datafile.lammps',
                },
                "_files_out":  {
                    "data_file"   :         system_name + '_nvtEquilibrated.lammps',
                },
                "system_name": system_name,
                "step"     :   "equilibration_nvt",
                "geninfo":     self.geninfo() # serves to distinguish duplicates
            },
            name="{:s}_equilibration_nvt".format(system_name))
        return fw

    def nptEquilibrate(self, system_name,
        lmp_suffix_template='-v baseName {baseName:s} -v dataFile {dataFile:s}'):
        """
        Sample for lmp_suffix: for a call like
            srun lmp -in lmp_equilibration_npt.input \
                -v has_indenter 1 -v pbc2d 0 -v reinitialize_velocities 1\
                -v baseName 377_SDS_on_AU_111_51x30x2_monolayer \
                -v dataFile 377_SDS_on_AU_111_51x30x2_monolayer.lammps
        set lmp_suffix='-v has_indenter 1 -v reinitialize_velocities 1 \
            -v baseName {baseName:s} -v dataFile {dataFile:s}'
        """

        get_input_files_ft = GetFilesTask( {
                'identifiers':    [ 'lmp_header.input', 'lmp_equilibration_npt.input', 'extract_thermo.sh' ] } )

        lmp_cmd = ' '.join((self._template_lmp_cmd.format(
            inputFile = 'lmp_equilibration_npt.input' ), lmp_suffix_template.format(
                baseName=system_name, dataFile='datafile.lammps' ) ))

        lmp_ft =  ScriptTask.from_str(
            lmp_cmd,
            {
                'stdout_file':  'lmp_nptEquilibration.out',
                'stderr_file':  'lmp_nptEquilibration.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        extract_thermo_ft =  ScriptTask.from_str(
            'bash extract_thermo.sh {:s} {:s}'.format(
                system_name + '_nptEquilibration.log',
                system_name + '_nptEquilibration_thermo.out'),
                {
                    'use_shell':    True, 'fizzle_bad_rc': False
                } )

        fw = Firework(
            [
                get_input_files_ft,
                lmp_ft,
                extract_thermo_ft
            ],
            spec={
                "_queueadapter": self.equilibration_queueadapter,
                "_category":     self.queue_worker,
                "_files_in":  {
                    "data_file"   :  'datafile.lammps',
                },
                "_files_out":  {
                    "data_file"   :         system_name + '_nptEquilibrated.lammps',
                },
                "system_name": system_name,
                "step"     :   "equilibration_npt",
                "geninfo":     self.geninfo() # serves to distinguish duplicates
            },
            name="{:s}_equilibration_npt".format(system_name))
        return fw

    def production(self, system_name,
        total_steps      = 5000000, # time steps, 5 mio ~ 10 ns
        lmp_suffix_template=' '.join((
            '-v baseName {baseName:s} -v dataFile {dataFile:s}',
            '-v has_indenter 0 -v pbc2d 0 -v mpiio 0 -v use_colvars 0')) ):
        """
        Sample for lmp_suffix: for a call like
            srun lmp -in lmp_production.input \
                -v has_indenter 0 -v pbc2d 0 -v mpiio 0 \
                -v thermo_frequency 1000 -v reinitialize_velocities 0 \
                -v use_colvars 0 -v productionSteps 1000 \
                -v baseName 377_SDS_on_AU_111_51x30x2_monolayer \
                -v dataFile 377_SDS_on_AU_111_51x30x2_monolayer.lammps
        set lmp_suffix='-v has_indenter 0 -v reinitialize_velocities 0 \
            -v baseName {baseName:s} -v dataFile {dataFile:s} ...'
        """
        get_input_files_ft = GetFilesTask( {
            'identifiers':    [
               'lmp_header.input',
               'lmp_production.input',
               'lmp_production_mixed.input',
               'extract_thermo.sh' ] } )

        lmp_cmd = ' '.join((self._template_lmp_cmd.format(
            inputFile = 'lmp_production.input' ), lmp_suffix_template.format(
                baseName=system_name, dataFile='datafile.lammps' ) ))

        lmp_ft =  ScriptTask.from_str(
            lmp_cmd,
            {
                'stdout_file':  'lmp_production.out',
                'stderr_file':  'lmp_production.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        extract_thermo_ft =  ScriptTask.from_str(
            'bash extract_thermo.sh {:s} {:s}'.format(
                system_name + '_production.log',
                system_name + '_production_thermo.out'),
                {
                    'use_shell':    True, 'fizzle_bad_rc': False
                } )

        # tail_ft_list = self.get_post_lammps_tasks( system_name,
        #    step = 'production', suffix = '_production', prefix = 'production_',
        #    lmp_suffix = '_production_mixed',
        #    output_folder = 'production_{:d}'.format(total_steps) )

        fw = Firework(
            [
                get_input_files_ft,
                lmp_ft,
                extract_thermo_ft
                #*tail_ft_list
            ],
            spec={
                "_queueadapter": self.production_queueadapter,
                "_category":     self.queue_worker,
                "_files_in":  {
                    "data_file"   :  'datafile.lammps',
                },
                "_files_out":  {
                    "data_file"   :     system_name + '_production.lammps',
                    "trajectory_file" : system_name + '_production_mixed.nc',
                    "thermo_file" :     system_name + '_production_thermo.out'
                },
                "system_name":    system_name,
                "step"     :      "production",
                "total_steps":    total_steps,
                "geninfo":        self.geninfo() # serves to distinguish duplicates
            },
            name="{:s}_produtcion_{:d}".format(system_name, total_steps) )
        return fw

    # TODO: parameters should appear in sim_df only
    def colvars_production(self, system_name,
        force_constant   = 2500.0, # kcal / mole
        total_steps      = 500000, # time steps for indenter approach
        initial_distance = 120.0, # nm, COM-COM
        final_distance   = 10.0, #nm, COM-COM
        lmp_suffix_template=' '.join((
            '-v baseName {baseName:s} -v dataFile {dataFile:s}',
            '-v has_indenter 1 -v pbc2d 0 -v mpiio 0 -v use_colvars 1')) ):
        """
        Sample for lmp_suffix: for a call like
            srun lmp -in lmp_production.input \
                -v has_indenter 1 -v pbc2d 0 -v mpiio 0 \
                -v thermo_frequency 1000 -v reinitialize_velocities 0 \
                -v use_colvars 1 -v productionSteps 1000 \
                -v baseName 377_SDS_on_AU_111_51x30x2_monolayer \
                -v dataFile 377_SDS_on_AU_111_51x30x2_monolayer.lammps
        set lmp_suffix='-v has_indenter 1 -v reinitialize_velocities 0 \
            -v baseName {baseName:s} -v dataFile {dataFile:s} ...'
        """

        get_input_files_ft = FileTransferTask( {
            'files': [
                {
                    'src':  self.template_prefix + os.sep + 'lmp_header.input',
                    'dest': 'lmp_header.input'
                }, {
                    'src':  self.template_prefix + os.sep + 'lmp_production.input',
                    'dest': 'lmp_production.input'
                }, {
                    'src':  self.template_prefix + os.sep + 'lmp_production_mixed.input',
                    'dest': 'lmp_production_mixed.input'
                } #,  {
                #    'src':  self.template_prefix + os.sep + 'colvars.inp',
                #    'dest': 'colvars.inp'
                # }
            ],
            'mode': 'copy' } )

        colvars_script_writer_task_context = {
            'header':         self.geninfo(),
            'output_frequency':            1,
            'restart_frequency':        1000,
            'lower_boundary':            0.0,
            'upper_boundary':          160.0,
            'force_constant':              force_constant,
            'initial_com_com_distance':  initial_distance,
            'final_com_com_distance':      final_distance,
            'total_steps':                    total_steps #  1ns
        }

        colvars_fill_script_template_ft = TemplateWriterTask( {
            'context' :      colvars_script_writer_task_context,
            'template_file': self.template_prefix + os.sep + 'colvars.inp',
            'output_file':   'colvars.inp' } )

        lmp_cmd = ' '.join((self._template_lmp_cmd.format(
            inputFile = 'lmp_production.input' ), lmp_suffix_template.format(
                baseName=system_name, dataFile='datafile.lammps' ) ))

        lmp_ft =  ScriptTask.from_str(
            lmp_cmd,
            {
                'stdout_file':  'lmp_production.out',
                'stderr_file':  'lmp_production.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        tail_ft_list = self.get_post_lammps_tasks( system_name,
            step = 'production', suffix = '_production', prefix = 'production_',
            lmp_suffix = '_production_mixed',
            output_folder = 'production_{:d}'.format(total_steps) )

        # insert removal of colvars output
        # extract_thermo_ft =  ScriptTask.from_str(
        #     self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
        #         system_name + '_production.log',
        #         system_name + '_production_thermo.out'),
        #         {
        #             'use_shell':    True, 'fizzle_bad_rc': False
        #         } )
        #
        # colvars_file_endings = [
        #     '*.colvars.traj',
        #     '*.colvars.state',
        #     '*.ti.count',
        #     '*.ti.grad',
        #     '*.ti.pmf'
        # ]
        #
        # tar_ft = ScriptTask.from_str(
        #     'tar -czf {:s} {:s}'.format(
        #         'colvars.tar.gz',
        #         ' '.join(colvars_file_endings) ),
        #     {
        #         'stdout_file':  'colvars_tar.out',
        #         'stderr_file':  'colvars_tar.err',
        #         'use_shell':    True,
        #         'fizzle_bad_rc':False
        #     })
        #
        # dest = self.output_prefix + os.sep + system_name + os.sep + 'production_{:d}'.format(total_steps)
        # mkdir_ft =  ScriptTask.from_str(
        #     'mkdir -p "{:s}"'.format(dest),
        #         {
        #             'use_shell':     True,
        #             'fizzle_bad_rc': False
        #         } )
        #
        # copy_output_files_ft = FileTransferTask( {
        #     'files': [
        #         system_name + '_production_mixed.lammps',
        #         system_name + '_production_mixed.nc',
        #         system_name + '_production.log',
        #         system_name + '_production_thermo.out',
        #         'lmp_production.out',
        #         'colvars.tar.gz'
        #     ],
        #     'dest': dest,
        #     'mode': 'copy' } )

        fw = Firework(
            [
                get_input_files_ft,
                colvars_fill_script_template_ft,
                lmp_ft,
                *tail_ft_list
                # extract_thermo_ft,
                # mkdir_ft,
                # tar_ft,
                # copy_output_files_ft
            ],
            spec={
                "_queueadapter": self.production_queueadapter,
                "_category":     self.queue_worker,
                "_dupefinder":   DupeFinderExact(),
                "_files_in":  {
                    "data_file"   :  'datafile.lammps',
                },
                "_files_out":  {
                    "data_file"   :  system_name + '_production.lammps',
                },
                "system_name":    system_name,
                "step"     :      "production",
                "force_constant": force_constant,
                "total_steps":    total_steps,
                "geninfo":        self.geninfo() # serves to distinguish duplicates
            },
            name="{:s}_produtcion_{:d}".format(system_name, total_steps) )
        return fw

    def get_post_lammps_tasks(self, system_name,
        step='production', suffix=None, prefix=None, lmp_suffix = None,
        output_folder = None):
        if suffix is None:
            suffix = '_' + step
        if prefix is None:
            prefix = step + '_'
        if lmp_suffix is None:
            lmp_suffix = suffix
        if output_folder is None:
            output_folder = step

        # insert removal of colvars output
        extract_thermo_ft =  ScriptTask.from_str(
            self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
                system_name + suffix + '.log',
                system_name + suffix + '_thermo.out'),
                {
                    'use_shell':    True, 'fizzle_bad_rc': False
                } )

        colvars_file_endings = [
            '*.colvars.traj',
            '*.colvars.state',
            '*.ti.count',
            '*.ti.grad',
            '*.ti.pmf'
        ]

        tar_ft = ScriptTask.from_str(
            'tar -czf {:s} {:s}'.format(
                'colvars.tar.gz',
                ' '.join(colvars_file_endings) ),
            {
                'stdout_file':  'colvars_tar.out',
                'stderr_file':  'colvars_tar.err',
                'use_shell':    True,
                'fizzle_bad_rc':False
            })

        dest = self.output_prefix + os.sep + system_name + os.sep + output_folder
        mkdir_ft =  ScriptTask.from_str(
            'mkdir -p "{:s}"'.format(dest),
                {
                    'use_shell':     True,
                    'fizzle_bad_rc': False
                } )

        copy_output_files_ft = FileTransferTask( {
            'files': [
                system_name + lmp_suffix + '.lammps',
                system_name + lmp_suffix + '.nc',
                system_name + suffix + '.log',
                system_name + suffix + '_thermo.out',
                'lmp' + suffix + '.out',
                'colvars.tar.gz'
            ],
            'dest':          dest,
            'mode':          'copy',
            'ignore_errors': True } )

        return [ extract_thermo_ft, tar_ft, mkdir_ft, copy_output_files_ft ]

    def post_lammps_fw(self, system_name, parent_fw_id, launch_dir = None,
        step='production', suffix=None, prefix=None, lmp_suffix = None,
        output_folder=None):
        if suffix is None:
            suffix = '_' + step
        if prefix is None:
            prefix = step + '_'
        if lmp_suffix is None:
            lmp_suffix = suffix
        if output_folder is None:
            output_folder = step

        fw_spec = {
            "_allow_fizzled_parents": True,
            "_category":     self.std_worker,
            "_dupefinder":   DupeFinderExact(),
            "_files_out":  {
                "data_file"   :  system_name + suffix + '.lammps',
            },
            "system_name":    system_name,
            "step"     :      'post_lammps',
            "geninfo":        self.geninfo() # serves to distinguish duplicates
        }
        if launch_dir is not None:
            fw_spec["_launch_dir"] = launch_dir

        fw = Firework(
            self.get_post_lammps_tasks( system_name,
            step, suffix, prefix, lmp_suffix, output_folder ),
            spec=fw_spec,
            name="{:s}_post_lammps".format(system_name))
        return fw

    def batch_prepare_fw( self, system_names, fw_creator ):
        fw_list = []
        for system_name in system_names:
            fw = fw_creator( system_name )
            fw_list.append((system_name,fw))

        fw_dict = dict(fw_list)
        return fw_dict

