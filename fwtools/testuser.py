#from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact
from fireworks.user_objects.firetasks.fileio_tasks import FileTransferTask, FileWriteTask
from fireworks.user_objects.firetasks.script_task import ScriptTask, PyTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask, GetFilesTask, DeleteFilesTask
from fireworks import Firework, LaunchPad, Workflow, FWorker, FireTaskBase, FWAction
import os
import scipy.constants as C
import numpy as np
from fireworks.user_objects.firetasks.jlh_tasks import MakeSegIdSegPdbDictTask
from fireworks.user_objects.firetasks.jlh_tasks import CmdTask

def run_replicate(dimensions,surfactants,sf_nmolecules,counterion,preassembly,sb_unit):
    inputs=get_inputs(dimensions,surfactants,sf_nmolecules,counterion,preassembly,sb_unit)
    replicate_fw=sb_replicate(inputs)
    return FWAction(stored_data={'inputs':inputs}, mod_spec=[{'_set': {'inputs': inputs}}], detours=replicate_fw)


def sb_replicate(inputs):
    dimensions=(inputs[0])["sb_multiples"]
    xyz=(dimensions[0])+' '+(dimensions[1])+' '+(dimensions[2])+' '
    sb_name=(inputs[0])["sb_name"]
    in_dimensions= xyz + '111' 
    cmd_run = ' '.join(('module load Fireworks; module load GROMACS/2018.1-gnu-7.3-openmpi-2.1.1; module load MDTools;', 'replicate.sh {:s}'.format(in_dimensions) ))
    replicate_ft =  ScriptTask.from_str( cmd_run ,{
                'use_shell':    True } )
    
    replicate_get_ft = GetFilesTask( {
                'identifiers': [ 'au_cell_P1_111.gro']} )
    replicate_set_ft = AddFilesTask({'compress':True ,'identifiers': ['{:s}.pdb'.format(sb_name)], 'paths': "{:s}.pdb".format(sb_name)})
    replicate_fw = Firework(
        [ replicate_get_ft, replicate_ft,replicate_set_ft
            ],
            spec={
                "_category":   "bwcloud_noqueue",
                "step"     :   "replicate",
            },
            name= 'replicate' )
    return replicate_fw

def get_inputs(dimensions,surfactants,sf_nmolecules,counterion,preassembly,sb_unit):
    sb_name= 'AU_111_'+dimensions[0]+'x'+dimensions[1]+'x'+dimensions[2]
    system_name=[sf_nmolecules+'_'+surfactants[i]+'_on_'+sb_name+'_'+preassembly[j]  for i in range(len(surfactants))  for j in range(len(preassembly))]
    sb_measures=np.asarray(list(map(int,dimensions)))*np.asarray(list(map(float,sb_unit)))
    box=[sb_measures[0],sb_measures[1],1.8e-08]
    sb_measures= list(map(str,sb_measures))
    inputs = [dict() for i in range(len(preassembly)*len(surfactants))]
   
    for i in range(len(surfactants)):
        for j in range(len(preassembly)):
            my_dict={
                       "sb_measures": sb_measures,
                       "sb_multiples":dimensions,
                       "sb_unit":sb_unit,
                       "surfactant":surfactants[i] ,
                       "preassembly":preassembly[j],
                       "sb_name":sb_name,
                       "sf_nmolecules":sf_nmolecules,
                       "counterion": counterion,
                       "system_name":system_name[i*len(preassembly)+j],
                       "box": box
                        }
            inputs[i*len(preassembly)+j]=my_dict
    return inputs

def run_packmol(inputs,nloop,maxit):
    packmol_wfs= ['' for x in range((len(inputs)))] 
    for i in range((len(inputs))):
        inputs_single=inputs[i]
        if "cylinders" in inputs_single["preassembly"]:
            packmol=prepare_packmol(inputs_single,pack_cylinders,nloop,maxit)
        elif "bilayer" in inputs_single["preassembly"]:
            packmol=prepare_packmol(inputs_single,pack_bilayer,nloop,maxit)
        elif "monolayer" in inputs_single["preassembly"]:
            packmol=prepare_packmol(inputs_single,pack_monolayer,nloop,maxit)
        packmol_wfs[i]=packmol
    return FWAction(detours=packmol_wfs)

def run_prepare_pdb2lmp(inputs):
    prepare_pdb2lmp_wfs= ['' for x in range((len(inputs)))] 
    for i in range((len(inputs))):
        inputs_single=inputs[i]
        prepare_pdb2lmp_wfs[i]=prepare_pdb2lmp(inputs_single)
    return FWAction(detours= prepare_pdb2lmp_wfs)


            

def prepare_packmol(inputs, pack_aggregates, nloop = None, maxit = None ):
        """Creates preassembled aggregates on surface"""
        #for system_name, row in sim_df.loc[system_names].iterrows():
        tolerance = 2 # Angstrom
        #No
        #
        
        surfactant  = inputs["surfactant"]
        counterion  = inputs["counterion"]
        sfN         = int(inputs["sf_nmolecules"])
        sb_name     = inputs["sb_name"]
        system_name=  inputs["system_name"]
        # measures of substrate:
        #No
        sb_measures=  list(map(float,inputs["sb_measures"]))
        sb_measures = np.asarray(sb_measures) / C.angstrom
        #
        # standard settings can be overridden
        #No
        packmol_script_writer_task_context = {
            'system_name':   system_name,
            'sb_name':       sb_name,
            'tolerance':     tolerance,
            'write_restart': True
        }
        #
        if nloop is not None:
            packmol_script_writer_task_context['nloop'] = nloop

        if maxit is not None:
            packmol_script_writer_task_context['maxit'] = maxit
        
        if "cylinders" in inputs["preassembly"]:
            if "hemicylinders" in inputs["preassembly"]:
                    packmol_script_writer_task_context.update(
                        pack_aggregates(
                            surfactant  = surfactant,
                            counterion  = counterion,
                            sfN = sfN,
                            sb_measures = sb_measures,
                            hemicylinders = True ))
            else:
                packmol_script_writer_task_context.update(
                    pack_aggregates(
                            surfactant  = surfactant,
                            counterion  = counterion,
                            sfN = sfN,
                            sb_measures = sb_measures,
                            hemicylinders = False
                            ))
        else:
            packmol_script_writer_task_context.update(
                pack_aggregates(
                    surfactant  = surfactant,
                    counterion  = counterion,
                    sfN = sfN,
                    sb_measures = sb_measures))
     
        packmol_get_template_ft = GetFilesTask( {
                'identifiers': [ 'packmol.inp' ]} )
        # packmol fill script template
        packmol_fill_script_template_ft = TemplateWriterTask( {
            'context' :      packmol_script_writer_task_context,
            #'template_file': '/mnt/dat/work/testuser/adsorption/N_surfactant_on_substrate_template/packmol.inp',
             'template_file': 'packmol.inp',
            'output_file':   system_name + '_packmol' + '.inp' } )

        packmol_fill_script_template_fw = Firework(
            [ packmol_get_template_ft,
            packmol_fill_script_template_ft],
            spec={
                "_category": "bwcloud_noqueue",
                '_files_out': {
                    'packmol_inp': system_name + '_packmol.inp'
                    },
                "system_name": system_name,
                "step"   :     "packmol_fill_script_template",
            },
            name="{:s}_packmol_fill_script_template".format(system_name) )

    # packmol get building blocks #database
        single_surfactant_pdb = '1_{:s}.pdb'.format(surfactant)
        single_counterion_pdb = '1_{:s}.pdb'.format(counterion)
        
        packmol_get_components_ft = GetFilesTask( {
                'identifiers': [ single_surfactant_pdb ,single_counterion_pdb ],
                'new_file_names': [ single_surfactant_pdb ,single_counterion_pdb] } )
        #database'
        packmol_get_substrate_ft = GetFilesTask( {
                'identifiers': [ '{:s}.pdb'.format(sb_name) ],
                'new_file_names': [ '{:s}.pdb'.format(sb_name) ] } )
    # packmol run
        infile= system_name + '_packmol.inp' 
        cmd_run = ' '.join(('module load MDTools;', 'packmol < {:s}'.format(infile) ))
        packmol_ft =  ScriptTask.from_str( cmd_run ,{
                'stdout_file':  system_name + '_packmol' + '.out',
                'stderr_file':  system_name + '_packmol' + '.err',
                'use_shell':    True,
                'shell_exe': "/bin/bash",
                'fizzle_bad_rc':True } )
        packmol_add_files_ft= AddFilesTask({'compress':True ,
                                            'identifiers': [system_name + '_packmol.inp', system_name + '_packmol.pdb'],
                                            'paths':[ system_name + '_packmol.inp',system_name + '_packmol.pdb']})
        
        packmol_fw = Firework(
            [
                packmol_get_components_ft,
                packmol_get_substrate_ft,
                packmol_ft,
                packmol_add_files_ft
            ],
            spec={
                "_category":   "bwcloud_noqueue",
                "_files_in" : {
                    "packmol_inp":  (system_name + '_packmol.inp') },
                "_files_out": {
                    "packmol_pdb" : (system_name + '_packmol.pdb'),
                    "packmol_pdb_FORCED" : (system_name + '_packmol.pdb_FORCED')
                },
                "system_name": system_name,
                "step"     :   "packmol",
            },
            name= system_name + '_packmol',
            parents=[packmol_fill_script_template_fw] )
        
        packmol_wf = Workflow(
            [packmol_fill_script_template_fw,
             packmol_fw
            ],
            {packmol_fill_script_template_fw: packmol_fw},
            name = system_name + '_packmol' )

        return packmol_wf


def pack_cylinders(sb_measures, surfactant, counterion, sfN, hemicylinders,ncylinders= 3):
    if surfactant=='SDS':
        l_surfactant = 14.0138 # Angstrom
        head_atom_number = 1
        tail_atom_number= 39
    elif surfactant=='CTAB':    
        l_surfactant = 19.934 # Angstrom
        head_atom_number= 17
        tail_atom_number= 1
        # Tolerance in packmol
    tolerance = 2 # Angstrom
        #NA or BR
    #counterion='NA'
    """Creates preassembled (hemi-) cylinders on substrate with couinterions at polar heads"""

    hemistr = 'hemi-' if hemicylinders else ''
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
    #  inner radius:            1*tolerance + l_surfactant
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

def pack_monolayer(sfN, sb_measures, surfactant, counterion ):
        """Creates preassembled monolayer"""

        if surfactant=='SDS':
            l_surfactant = 14.0138 # Angstrom
            head_atom_number = 1
            tail_atom_number= 39
        elif surfactant=='CTAB':    
            l_surfactant = 19.934 # Angstro
            head_atom_number= 17
            tail_atom_number= 1
        
        tolerance = 2 # Angstrom
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

def pack_bilayer(sfN, sb_measures, surfactant, counterion):
        """Creates a single bilayer on substrate with couinterions at polar heads"""
        
        if surfactant=='SDS':
            l_surfactant = 14.0138 # Angstrom
            head_atom_number = 1
            tail_atom_number= 39
        elif surfactant=='CTAB':    
            l_surfactant = 19.934 # Angstro
            head_atom_number= 17
            tail_atom_number= 1
        
        tolerance = 2 # Angstrom
        
        
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

def get_dimensions():
    ini=0
    fin=0
    path=os.getcwd()
    #path='/mnt/dat/work/testuser/adsorption/N_surfactant_on_substrate_template'
    files = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and 'AU_111' in i:
            files.append(i)
    s=list(files[0])
    count=0
    for i in range(len(s)):
        if s[i]=='1':
            count+=1
        if count==3:
            ini=i
            count=0
            break
    for i in range(ini,len(s)):
        if s[i]=='_':
            count+=1
        if count==2:
            fin=i
            break
    result=(''.join(s[ini+2:fin]))
    return result

def get_surfactants(dimensions,surfactants):
    outputs=[surfactants[i]+'_on_AU_111_'+ dimensions  for i in range(len(surfactants))]
    return outputs

def get_preassembly(dim_surf,preassembly):
    for i in range(len(dim_surf)):
        outputs=[dim_surf[i]+'_'+ preassembly[j] for j in range(len(preassembly))]
    return outputs

def get_inputs_single(dim_surf_asb, sf_nmolecules, counterion, sb_unit):
    dim_surf_asb=dim_surf_asb[0]
    split_name=dim_surf_asb.split('_')
    surfactants=[split_name[0]]
    dimensions=split_name[4].split('x')
    preassembly=['_'.join((split_name[5:]))]
    sb_name= 'AU_111_'+dimensions[0]+'x'+dimensions[1]+'x'+dimensions[2]
    system_name=[sf_nmolecules+'_'+surfactants[i]+'_on_'+sb_name+'_'+preassembly[j]  for i in range(len(surfactants))  for j in range(len(preassembly))]
    sb_measures=np.asarray(list(map(int,dimensions)))*np.asarray(list(map(float,sb_unit)))
    sb_measures= list(map(str,sb_measures))
    inputs = [dict() for i in range(len(preassembly)*len(surfactants))]
    for i in range(len(surfactants)):
        for j in range(len(preassembly)):
            my_dict={
                       "sb_measures": sb_measures,
                       "sb_multiples":dimensions,
                       "sb_unit":sb_unit,
                       "surfactant":surfactants[i] ,
                       "preassembly":preassembly[j],
                       "sb_name":sb_name,
                       "sf_nmolecules":sf_nmolecules,
                       "counterion": counterion,
                       "system_name":system_name[i*len(preassembly)+j],
                       "box": box}
            inputs[i*len(preassembly)+j]=my_dict
    return inputs

def prepare_pdb2lmp(inputs):
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

        sb_name    = inputs["sb_name"]

        surfactant = inputs["surfactant"]
        sfN        = inputs["sf_nmolecules"]
        system_name =inputs["system_name"]
        # make more flexible at some point
        if surfactant == 'SDS': # surfactant is anionic
            nanion = 0
            ncation = sfN
        else: # CTAB (surfactant is cationic)
            nanion = sfN
            ncation = 0

        box_nanometer = np.asarray( inputs["box"] ) / C.nano
        box_angstrom  = np.asarray( inputs["box"] ) / C.angstrom

        consecutive_fw_list = []

    # pdb to gro
         #
    
        packmol2gmx_get_files_ft = GetFilesTask( {
                'identifiers':  [  system_name + '_packmol' + '.pdb' ] } )
        
        packmol2gmx_get_files_fw = Firework(  packmol2gmx_get_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_out":  { "packmol_pdb":  system_name + '_packmol' + '.pdb' },
                "step"   :     "packmol2gmx_get_files",
                },
            name=(system_name + "packmol2gmx_get_files") 
        )
        consecutive_fw_list.append(packmol2gmx_get_files_fw)
        #
        
        packmol2gmx_cmd = ' '.join((
            'module load mdtools;',
            'pdb_packmol2gmx.sh '+ system_name + '_packmol' + '.pdb' ))
        
        packmol2gmx_ft =  ScriptTask.from_str(
                packmol2gmx_cmd,
            {
                'stdout_file':  system_name + '_packmol2gmx' + '.out',
                'stderr_file':  system_name + '_packmol2gmx' + '.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })
        
        packmol2gmx_fw = Firework( packmol2gmx_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "_files_in": {"packmol_pdb" : system_name + '_packmol' + '.pdb'},
                "system_name" : system_name,
                "_files_out":  { "pdb_for_gmx":  system_name + '.pdb' },
                "step"   :     "packmol2gmx",
                },
            name=(system_name + '_packmol2gmx'),
            parents=[packmol2gmx_get_files_fw]
        )
        consecutive_fw_list.append(packmol2gmx_fw)
        
        #
    
        packmol2gmx_add_files_ft = AddFilesTask( {
                'compress':True , 'identifiers':  [  system_name + '.pdb' ], 
                'paths': [system_name + '.pdb']} )
        
        packmol2gmx_add_files_fw = Firework(  packmol2gmx_add_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_in":  { "pdb_for_gmx":  system_name + '.pdb' },
                "step"   :     "packmol2gmx_add_files",
                },
            name=(system_name + "packmol2gmx_add_files"),
            parents=[packmol2gmx_fw]
       )
        consecutive_fw_list.append(packmol2gmx_add_files_fw)
        # 

    # solvate in gromacs:
        #
        gmx_solvate_get_files_ft = GetFilesTask( {
                'identifiers': ['1_{:s}.pdb'.format(surfactant), system_name + '.pdb'] } )
        
        gmx_solvate_get_files_fw = Firework(gmx_solvate_get_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_out":  { "surfactant_pdb":  '1_{:s}.pdb'.format(surfactant),
                                "pdb_for_gmx":system_name + '.pdb' },
                "step"   :     "gmx_solvate_get_files",
                },
            name=(system_name + "gmx_solvate_get_files"),
            parents=[packmol2gmx_add_files_fw]
        )
        consecutive_fw_list.append(gmx_solvate_get_files_fw)
        
        #
        gmx_solvate_fill_script_template_ft = TemplateWriterTask( {
            'context': {
                'system_name':system_name,
                'surfactant': surfactant,
                'ncation':    ncation,
                'nanion':     nanion,
                'box':        box_nanometer,
                'ionize':     False
            },
            'template_file': 'gmx_solvate.sh',
            'output_file':   system_name + '_gmx_solvate' + '.sh'} )

        

        template_gmx2pdb_cmd= ' '.join(('module load gromacs/2018.1-gnu-5.2;',
                                       'bash '+system_name+'_gmx_solvate'+'.sh'))
                                       
        gmx_solvate_ft =  ScriptTask.from_str(template_gmx2pdb_cmd,
                                             {'stdout_file':  system_name + '_gmx_solvate' + '.out',
                                              'stderr_file':  system_name + '_gmx_solvate' + '.err',
                                              'use_shell':    True,
                                              'fizzle_bad_rc':True} )
    
        gmx_solvate_fw = Firework(
            [
                gmx_solvate_fill_script_template_ft,
                gmx_solvate_ft,
            ],
            spec={
                "_category":  "nemo_noqueue",
                "system_name": system_name,
                "_files_in":   { "surfactant_pdb":  '1_{:s}.pdb'.format(surfactant),
                                "pdb_for_gmx":system_name + '.pdb'},
                "_files_out":  {
                    "ionized_gro" : "{:s}_solvated.gro".format(system_name)
                    },
                "step"   :     "gmx_solvate",
            },
            name="{:s}_gmx_solvate".format(system_name),
            parents=[gmx_solvate_get_files_fw])

        consecutive_fw_list.append(gmx_solvate_fw)   
        
        gmx_solvate_add_files_ft = AddFilesTask( {
                'compress':True , 'identifiers':  [ "{:s}_solvated.gro".format(system_name) ]  ,
                'paths': ["{:s}_solvated.gro".format(system_name)]} )
        
        gmx_solvate_add_files_fw = Firework(gmx_solvate_add_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_in":  { "ionized_gro" : "{:s}_solvated.gro".format(system_name) },
                "step"   :     "gmx_solvate_add_files",
                },
            name=(system_name + "gmx_solvate_add_files") ,
            parents=[gmx_solvate_fw])
        consecutive_fw_list.append(gmx_solvate_add_files_fw)
    

    # convert to pdb chunks again
        gmx2pdb_get_files_ft = GetFilesTask( {
                'identifiers': [  "{:s}_solvated.gro".format(system_name)],
                'new_file_names': [  "{:s}_ionized.gro".format(system_name)]})
        
        gmx2pdb_get_files_fw = Firework(gmx2pdb_get_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_out":  { "ionized_gro" : "{:s}_ionized.gro".format(system_name) },
                "step"   :     "gmx2pdb_get_files",
                },
            name=(system_name + "gmx2pdb_get_files") ,
            parents=[gmx_solvate_add_files_fw]
        )
        consecutive_fw_list.append(gmx2pdb_get_files_fw)
      
    
        pdb_segment_chunk_glob_pattern = '*_[0-9][0-9][0-9].pdb'

        gmx2pdb_fill_script_template_ft = TemplateWriterTask( {
            'context': {
                'system_name':  system_name,
            },
            'template_file': 'gmx2pdb.sh',
            'output_file': system_name + '_gmx2pdb' + '.sh'} )

        infile=' '.join((
            'module load gromacs/2018.1-gnu-5.2 vmd;',
            'source '+system_name + '_gmx2pdb' + '.sh' ))        
        gmx2pdb_ft =  ScriptTask.from_str((infile),
            {
                'stdout_file':  system_name + '_gmx2pdb' + '.out',
                'stderr_file':  system_name + '_gmx2pdb' + '.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        gmx2pdb_tar_ft = ScriptTask.from_str(
            'tar -czf {:s} {:s}'.format(
                system_name + '_segments.tar.gz',
                pdb_segment_chunk_glob_pattern ),
            {
                'stdout_file':  system_name + '_gmx2pdb' + '_tar.out',
                'stderr_file':  system_name + '_gmx2pdb' + '_tar.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })
                ## check

        gmx2pdb_fw = Firework(
            [   gmx2pdb_fill_script_template_ft,
                gmx2pdb_ft,
                gmx2pdb_tar_ft
            ],
            spec={
                "_category":   "nemo_noqueue",
                "system_name" :system_name,
                "_files_in":   {
                    "ionized_gro" : "{:s}_ionized.gro".format(system_name)
                },
                "_files_out":  {
                    "ionized_gro" : "{:s}_ionized.gro".format(system_name),
                    "segments_tar": "{:s}_segments.tar.gz".format(system_name)
                                },
                "step"   :     "gmx2pdb"
            },
            name="{:s}_gmx2pdb".format(system_name),
            parents=[gmx2pdb_get_files_fw])

        consecutive_fw_list.append(gmx2pdb_fw)         

    ### PSFGEN
    
    
    #
        psfgen_get_files_ft = GetFilesTask( {
                'identifiers':    ['par_all36_lipid_extended_stripped.prm', 
                                   'top_all36_lipid_extended_stripped.rtf' ] } )
    
    
        psfgen_get_files_fw = Firework(psfgen_get_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_in": {"segments_tar":"{:s}_segments.tar.gz".format(system_name)},
                "_files_out": {"par_all36":'par_all36_lipid_extended_stripped.prm',
                               "top_all36":'top_all36_lipid_extended_stripped.rtf',
                               "tar_gz":"{:s}_segments.tar.gz".format(system_name)},
                "step"   :     "psfgen_get_files",
                },
            name=(system_name + "psfgen_get_files") ,
            parents=[gmx2pdb_fw]
        )
        consecutive_fw_list.append(psfgen_get_files_fw)
      
    #

    # make psfgen input
        psfgen_untar_tf = ScriptTask.from_str('tar -xf {:s}'.format(
            system_name + '_segments.tar.gz'),
            {
                'stdout_file':  system_name + '_psfgen' + '_untar.out',
                'stderr_file':  system_name + '_psfgen' + '_untar.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        makeIdPdbDict_ft = MakeSegIdSegPdbDictTask( {
            'glob_pattern': pdb_segment_chunk_glob_pattern } )

    # get necessary files
    ## check
        psfgen_fill_script_template_ft = TemplateWriterTask( {
            'use_global_spec' : True} )

    # run psfgen

        template_psfgen_cmd  = ' '.join((
            'module load vmd;',
            'vmd -e '+ system_name + '_psfgen'+ '.pgn' ))
        
        psfgen_ft =  ScriptTask.from_str((template_psfgen_cmd ),
            {
                'stdout_file':  system_name + '_psfgen' + '.out',
                'stderr_file':  system_name + '_psfgen'+ '.err',
                'use_shell':    True,
                'fizzle_bad_rc':True
            })

        psfgen_fw = Firework(
            [
                psfgen_untar_tf,
                makeIdPdbDict_ft,
                psfgen_fill_script_template_ft,
                psfgen_ft
            ],
            spec={
                "_category":  "nemo_noqueue",
                "system_name": system_name,
                "_files_in":  {
                    'par_all36':'par_all36_lipid_extended_stripped.prm', 
                    'top_all36':'top_all36_lipid_extended_stripped.rtf',
                    'tar_gz':"{:s}_segments.tar.gz".format(system_name)},
                "_files_out": {
                    'par_all36':'par_all36_lipid_extended_stripped.prm',
                    'top_all36':'top_all36_lipid_extended_stripped.rtf',                    
                    "psfgen_pdb": "{:s}_psfgen.pdb".format(system_name),
                    "psfgen_psf": "{:s}_psfgen.psf".format(system_name),
                    "psfgen_pgn": "{:s}_psfgen.pgn".format(system_name)},
                "step"       : "psfgen",
                # for the template writer task
                'context': {
                    'system_name': system_name
                },
                'template_file': 'psfgen.pgn',
                'output_file':   system_name + '_psfgen' + '.pgn'
            },
            name = system_name +'_psfgen',
            parents=[psfgen_get_files_fw])

        consecutive_fw_list.append(psfgen_fw)
        
        #
        psfgen_add_files_ft=AddFilesTask({'compress':True ,'identifiers':["{:s}_psfgen.pdb".format(system_name), 
                                                                          "{:s}_psfgen.psf".format(system_name), 
                                                                          system_name + '_psfgen' + '.pgn'],
                                           'paths': ["{:s}_psfgen.pdb".format(system_name), 
                                                     "{:s}_psfgen.psf".format(system_name),
                                                     system_name + '_psfgen' + '.pgn']})
            
        psfgen_add_files_fw = Firework( psfgen_add_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_in":  {
                    'par_all36':'par_all36_lipid_extended_stripped.prm',
                    'top_all36':'top_all36_lipid_extended_stripped.rtf', 
                    "psfgen_pdb": "{:s}_psfgen.pdb".format(system_name),
                    "psfgen_psf":  "{:s}_psfgen.psf".format(system_name), 
                    "psfgen_pgn":  system_name + '_psfgen' + '.pgn'},
                "_files_out":  {
                    'par_all36':'par_all36_lipid_extended_stripped.prm',
                    'top_all36':'top_all36_lipid_extended_stripped.rtf', 
                    "psfgen_pdb": "{:s}_psfgen.pdb".format(system_name),
                    "psfgen_psf": "{:s}_psfgen.psf".format(system_name),
                    "psfgen_pgn":  system_name + '_psfgen' + '.pgn'},

                "step"   :     "psfgen_add_files",
                },
            name=(system_name + "psfgen_add_files") ,
            parents=[psfgen_fw]
        )
        consecutive_fw_list.append(psfgen_add_files_fw)
        
        #
        

    # run charmm2lammps

       # ch2lmp_get_files_ft = GetFilesTask( {
       #        'identifiers':    ['par_all36_lipid_extended_stripped.prm', 
       #                          'top_all36_lipid_extended_stripped.rtf',
       #                            "{:s}_psfgen.pdb".format(system_name), 
       #                            "{:s}_psfgen.psf".format(system_name) ] } )
        
        
       # ch2lmp_get_files_fw = Firework(ch2lmp_get_files_ft,
       #     spec = {
       #         "_category":  "nemo_noqueue",
       #         "system_name" : system_name,

       #         "_files_out": {"par_all36":'par_all36_lipid_extended_stripped.prm', 
       #                        "top_all36":'top_all36_lipid_extended_stripped.rtf', 
       #                        "psfgen_pdb":"{:s}_psfgen.pdb".format(system_name),
       #                        "psfgen_psf":"{:s}_psfgen.psf".format(system_name)},
       #         "step"   :     "ch2lmp_get_files",
       #         },
       #     name=(system_name + "ch2lmp_get_files") ,
       #     parents=[psfgen_add_files_fw]
       #)
       # consecutive_fw_list.append(ch2lmp_get_files_fw)  
                 
        
        template_ch2lmp_cmd  = ' '.join((
            'module load mdtools;',
            'charmm2lammps.pl all36_lipid_extended_stripped {system_name:s}_psfgen',
            '-border=0 -lx={box[0]:.3f} -ly={box[1]:.3f} -lz={box[2]:.3f} ' ))
        
        
        ch2lmp_ft =  ScriptTask.from_str(template_ch2lmp_cmd.format(
                    system_name = system_name, box = box_angstrom), {
                'stdout_file':  system_name + '_ch2lmp' + '.out',
                'stderr_file':  system_name + '_ch2lmp' + '.err',
                'use_shell':    True,
                'fizzle_bad_rc':True } )

        # If existent, just deletes existing files
        file_identifier = "{:s}_psfgen.data".format(system_name)

        #ch2lmp_delete_ft = DeleteFilesTask(
        #    {
        #        'identifiers': [file_identifier]
        #    }
        #)


        ch2lmp_fw = Firework(
            [
                ch2lmp_ft
            ],
            spec={
                "_category":   "nemo_noqueue",
                "system_name": system_name,
                "_files_in": {
                    "par_all36":'par_all36_lipid_extended_stripped.prm',
                    "top_all36":'top_all36_lipid_extended_stripped.rtf',
                    "psfgen_pdb": "{:s}_psfgen.pdb".format(system_name),
                    "psfgen_psf": "{:s}_psfgen.psf".format(system_name) },
                "_files_out": {
                    "ch2lmp_data": "{:s}_psfgen.data".format(system_name),
                    "ch2lmp_in": "{:s}_psfgen.in".format(system_name),
                    "ch2lmp_ctrl_pdb": "{:s}_psfgen_ctrl.pdb".format(system_name),
                    "ch2lmp_ctrl_psf": "{:s}_psfgen_ctrl.psf".format(system_name)},
                "step"       : "ch2lmp",
            },
            name= system_name + '_ch2lmp',
            parents=[psfgen_add_files_fw])

        consecutive_fw_list.append(ch2lmp_fw)
                
        ch2lmp_store_ft =  AddFilesTask(
            {
                'paths':       ["{:s}_psfgen.data".format(system_name),
                                 "{:s}_psfgen.in".format(system_name),
                                 "{:s}_psfgen_ctrl.pdb".format(system_name),
                                 "{:s}_psfgen_ctrl.psf".format(system_name)
                                  ],

                'identifiers': [ file_identifier,
                                 "{:s}_psfgen.in".format(system_name),
                                 "{:s}_psfgen_ctrl.pdb".format(system_name),
                                 "{:s}_psfgen_ctrl.psf".format(system_name)
                                 ],
                'metadata': {
                    'system_name':          inputs["system_name"],
                    'sb_name':              inputs["sb_name"],
                    'surfactant':           inputs["surfactant"],
                    #'substrate':            inputs["substrate"],
                    'counterion':           inputs["counterion"],
                    #'solvent':              inputs["solvent"],
                    'sf_preassembly':       inputs["preassembly"],
                    #'ci_preassembly':       inputs["ci_initial_placement"],
                    #'sb_crystal_plane':     inputs["sb_crystal_plane"]
                }
            }
        )
        
        ch2lmp_store_fw = Firework( ch2lmp_store_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "step"   :     "ch2lmp_store",
                "_files_in": {
                    "ch2lmp_data": "{:s}_psfgen.data".format(system_name),
                    "ch2lmp_in": "{:s}_psfgen.in".format(system_name),
                    "ch2lmp_ctrl_pdb": "{:s}_psfgen_ctrl.pdb".format(system_name),
                    "ch2lmp_ctrl_psf": "{:s}_psfgen_ctrl.psf".format(system_name)}
                },
            name=(system_name + "ch2lmp_store") ,
            parents=[ch2lmp_fw]
        )
        consecutive_fw_list.append(ch2lmp_store_fw )

        parent_links = { consecutive_fw_list[i] : consecutive_fw_list[i+1] \
                        for i in range(len(consecutive_fw_list)-1) }

        return Workflow( consecutive_fw_list, parent_links,
            name="{:s}_prep_wf".format(system_name) )

 
def run_prepare_pdb2lmp(inputs):
    prepare_pdb2lmp_wfs= ['' for x in range((len(inputs)))] 
    for i in range((len(inputs))):
        inputs_single=inputs[i]
        prepare_pdb2lmp_wfs[i]=prepare_pdb2lmp(inputs_single)
    return FWAction(detours= prepare_pdb2lmp_wfs)

def minimize(system_name,
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
        
        consecutive_fw_list = []

        get_input_files_ft = GetFilesTask( {
                'identifiers':    [#'lmp_header.input', 
                                   'lmp_header_v2.input',
                                   #'lmp_minimization.input',
                                   'lmp_minimization_v2.input',
                                   'extract_thermo.sh',
                                   system_name+'_psfgen.data' ] ,
                'new_file_names': ['lmp_header.input',
                                   'lmp_minimization.input',
                                   'extract_thermo.sh',
                                   'datafile.lammps']} )
        
        
        get_input_files_fw = Firework( get_input_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_out":  {"lmp_header":'lmp_header.input', 
                                "lmp_minimization":'lmp_minimization.input', 
                                "extract_thermo":'extract_thermo.sh', 
                               "datafile": 'datafile.lammps'},
                "step"   :     "get_files",
                },
            name=(system_name + "get_files") 
        )
        consecutive_fw_list.append(get_input_files_fw)
        
        
        lmp_cmd= ' '.join((
            'module load lammps/08May19-git-master-gnu-7.3-openmpi-3.1-colvars-08May19;',
            'mpirun -n 4 lmp -in '+ 'lmp_minimization.input', lmp_suffix_template.format(
                baseName= system_name, dataFile='datafile.lammps' ) ))

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
                'log.lammps',
                system_name + '_minimization_thermo.out'),
               {
                    'use_shell':    True, 'fizzle_bad_rc': False
                } )

        fw = Firework(
            [
                minimization_ft,
                extract_thermo_ft
            ],
            spec={
                "_category":    "nemo_queue_offline",
                "_queueadapter": {"nodes": 8,
                                  "ppn": 20,
                                  "queue": 'nemo_queue_offline',
                                  "walltime": "96:00:00"},
                "_files_in":  { "lmp_header":'lmp_header.input',
                                "lmp_minimization":'lmp_minimization.input', 
                                "extract_thermo":'extract_thermo.sh', 
                                "datafile": 'datafile.lammps'},
                "_files_out":  {"lmp_minimized":system_name + '.lammps',
                                "thermo_out":system_name + "_minimization_thermo.out"},
                "system_name": system_name,
                "step"     : "minimization"
            },
            name="{:s}_minimzation".format(system_name),
            parents=[get_input_files_fw] )
        
        consecutive_fw_list.append(fw)
        
        add_output_files_ft= AddFilesTask({'compress':True ,
                                           'identifiers': [ system_name + '_minimized.lammps',
                                                           system_name + "_minimization_thermo.out"], 
                                           'paths': [ system_name + '.lammps',
                                                      system_name + "_minimization_thermo.out"]})
        
        
        add_output_files_fw = Firework(add_output_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_in":  { "lmp_minimized":system_name + '.lammps',
                                "thermo_out":system_name + "_minimization_thermo.out"},
                "step"   :     "add_output_files",
                },
            name=(system_name + "add_output_files"),
            parents=[fw]
       )
        consecutive_fw_list.append(add_output_files_fw)
        
        parent_links = { consecutive_fw_list[i] : consecutive_fw_list[i+1] \
                        for i in range(len(consecutive_fw_list)-1) }

        return Workflow( consecutive_fw_list, parent_links,
            name="{:s}_min_wf".format(system_name) )

    
def nvtEquilibrate(system_name,
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

        consecutive_fw_list = []
        
        get_input_files_ft = GetFilesTask( {
                'identifiers':    ['lmp_header.input', 
                                   #'lmp_equilibration_nvt.input', 
                                   'lmp_equilibration_nvt_v2.input',
                                   'extract_thermo.sh', 
                                   system_name + '_minimized.lammps'
                                  ],
                'new_file_names': ['lmp_header.input', 
                                   'lmp_equilibration_nvt.input',
                                    'extract_thermo.sh',
                                   'datafile.lammps']} )
        
        get_input_files_fw = Firework( get_input_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_out":  {"lmp_header":'lmp_header.input', 
                                "lmp_equilibration":'lmp_equilibration_nvt.input', 
                                "extract_thermo":'extract_thermo.sh', 
                                "datafile": 'datafile.lammps'},
                "step"   :     "get_files",
                },
            name=(system_name + "get_files") 
        )
        consecutive_fw_list.append(get_input_files_fw)
  
        lmp_cmd = ' '.join((
            'module load lammps/08May19-git-master-gnu-7.3-openmpi-3.1-colvars-08May19;',
            'mpirun -n 4 lmp -in '+ 'lmp_equilibration_nvt.input' , lmp_suffix_template.format(
                baseName= system_name, dataFile='datafile.lammps' ) ) )

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

        fw = Firework(
            [
                lmp_ft,
                extract_thermo_ft
            ],
            spec={
             
                "_category":    "nemo_queue_offline",
                "_queueadapter": {"nodes": 8,
                                  "ppn": 20,
                                  "queue": 'nemo_queue_offline',
                                  "walltime": "96:00:00"},
                "system_name": system_name,
                "_files_in":  { "lmp_header":'lmp_header.input', 
                                "lmp_equilibration":'lmp_equilibration_nvt.input', 
                                "extract_thermo":'extract_thermo.sh', 
                                "datafile": 'datafile.lammps'},
                "_files_out":  {"lmp_nvt": system_name + '_nvt_Equilibrated.lammps',
                                "thermo_out": system_name + '_nvtEquilibration_thermo.out'},
                "step"     :   "equilibration_nvt",
            },
            name="{:s}_equilibration_nvt".format(system_name),
            parents=[get_input_files_fw] )
        
        
        consecutive_fw_list.append(fw)
    
        add_output_files_ft= AddFilesTask({'compress':True ,
                                           'identifiers': [ system_name + '_nvtEquilibrated.lammps'],
                                           'paths': [ system_name + '_nvtEquilibraed.lammps']})
        
        
        add_output_files_fw = Firework(add_output_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_in":  { "lmp_nvt": system_name + '_nvtEquilibrated.lammps',
                                "thermo_out": system_name + '_nvtEquilibration_thermo.out'},
                "step"   :     "add_output_files",
                },
            name=(system_name + "add_output_files"),
            parents=[fw]
       )
        consecutive_fw_list.append(add_output_files_fw)
        
        parent_links = { consecutive_fw_list[i] : consecutive_fw_list[i+1] \
                        for i in range(len(consecutive_fw_list)-1) }

        return Workflow( consecutive_fw_list, parent_links,
            name="{:s}_nvt_wf".format(system_name) )
    


def nptEquilibrate(system_name,
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
        consecutive_fw_list = []

        get_input_files_ft = GetFilesTask( {
                'identifiers':    ['lmp_header.input', 
                                   #'lmp_equilibration_npt.input',
                                   'lmp_equilibration_npt_v2.input', 
                                   'extract_thermo.sh',
                                   system_name + '_nvtEquilibrated.lammps'
                                  ],
                'new_file_names': ['lmp_header.input', 
                                   'lmp_equilibration_npt.input', 
                                   'extract_thermo.sh',
                                   'datafile.lammps']} )
        
        get_input_files_fw = Firework( get_input_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_queueadapter": {"nodes": 8,
                                  "ppn": 20,
                                  "queue": 'nemo_queue_offline',
                                  "walltime": "96:00:00"},
                "_files_out":  {"lmp_header":'lmp_header.input', 
                                "lmp_equilibration":'lmp_equilibration_npt.input', 
                                "extract_thermo":'extract_thermo.sh',
                                "datafile": 'datafile.lammps'},
                "step"   :     "get_files",
                },
            name=(system_name + "get_files")
                                      
        )
        consecutive_fw_list.append(get_input_files_fw)
        
        lmp_cmd = ' '.join((
            'module load lammps/08May19-git-master-gnu-7.3-openmpi-3.1-colvars-08May19;',
            'mpirun -n 4 lmp -in '+ 'lmp_equilibration_npt.input' , lmp_suffix_template.format(
                baseName= system_name, dataFile='datafile.lammps' ) ) )

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
                lmp_ft,
                extract_thermo_ft
            ],
            spec={
                "_category":   "nemo_queue_offline",
                "system_name": system_name,
                "step"     :   "equilibration_npt",
                "_files_in":  {"lmp_header":'lmp_header.input', 
                                "lmp_equilibration":'lmp_equilibration_npt.input', 
                                "extract_thermo":'extract_thermo.sh',
                                "datafile": 'datafile.lammps'},
                "_files_out": {"lmp_npt":  system_name + '_nptEquilibrated.lammps',
                               "thermo_out": system_name + '_nptEquilibration_thermo.out'}
            },
            name="{:s}_equilibration_npt".format(system_name),
            parents=[get_input_files_fw])
        
        consecutive_fw_list.append(fw)
        
        add_output_files_ft= AddFilesTask({'compress':True ,
                                           'identifiers': [ system_name + '_nptEquilibrated.lammps'], 
                                           'paths': [ system_name + '_nptEquilibrated.lammps']})
        
        add_output_files_fw = Firework(add_output_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_in":  { "lmp_npt": system_name + '_nptEquilibrated.lammps',
                                "thermo_out": system_name + '_nptEquilibration_thermo.out'},
                "step"   :     "add_output_files",
                },
            name=(system_name + "add_output_files"),
            parents=[fw]
       )
        consecutive_fw_list.append(add_output_files_fw)
        
        parent_links = { consecutive_fw_list[i] : consecutive_fw_list[i+1] \
                        for i in range(len(consecutive_fw_list)-1) }

        return Workflow( consecutive_fw_list, parent_links,
            name="{:s}_npt_wf".format(system_name) )

def production(system_name,
        total_steps      = 5000000, 
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
        consecutive_fw_list = []
        
        get_input_files_ft = GetFilesTask( {'identifiers':    [#'lmp_header.input',
                                                               'lmp_header_v2.input',
                                                               #'lmp_production.input',
                                                               'lmp_production_v2.input',
                                                               'extract_thermo.sh' , 
                                                               system_name + '_nptEquilibrated.lammps'],
                                            'new_file_names': ['lmp_header.input',
                                                               'lmp_production.input',
                                                               'extract_thermo.sh',
                                                               'datafile.lammps']} )
        
        get_input_files_fw = Firework( get_input_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_out":  {"lmp_header":'lmp_header.input', 
                                "lmp_production":'lmp_production.input', 
                                "extract_thermo":'extract_thermo.sh',
                                "datafile": 'datafile.lammps'},
                "step"   :     "get_files",
                },
            name=(system_name + "get_files")
                                      
        )
        consecutive_fw_list.append(get_input_files_fw)
        
        lmp_cmd = ' '.join((
            'module load lammps/08May19-git-master-gnu-7.3-openmpi-3.1-colvars-08May19;',
            'mpirun -n 16 lmp -in '+ 'lmp_production.input' , lmp_suffix_template.format(
                baseName= system_name, dataFile='datafile.lammps' ) ) )

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
                system_name+'_production.log',
                system_name + '_production_thermo.out'),
                {
                    'use_shell':    True, 'fizzle_bad_rc': False
                } )

        fw = Firework(
            [
                lmp_ft,
                extract_thermo_ft
                            ],
            spec={
                "_category":     "nemo_queue_offline",
                "_queueadapter": {"nodes": 8,
                                  "ppn": 20,
                                  "queue": 'nemo_queue_offline',
                                  "walltime": "96:00:00"},
                "system_name":    system_name,
                "_files_in":  {"lmp_header":'lmp_header.input', 
                               "lmp_production":'lmp_production.input', 
                               "extract_thermo":'extract_thermo.sh',
                               "datafile": 'datafile.lammps'},
                "step"     :   "production",
                "total_steps":    total_steps
            },
            name="{:s}_produtcion_{:d}".format(system_name, total_steps) ,
            parents=[get_input_files_fw])
        
        consecutive_fw_list.append(fw)


        add_output_files_ft= AddFilesTask({'compress': True,
                                           'identifiers': [ system_name + '_production_thermo.out'],
                                           'paths': [ system_name + '_production_thermo.out']})

        add_output_files_fw = Firework(add_output_files_ft,
            spec = {
                "_category": "nemo_noqueue",
                "system_name" : system_name,
                "_files_in": {"thermo_out": system_name + '_production_thermo.out'},
                "step"   : "add_output_files",
                },
            name=(system_name + "add_output_files"),
            parents=[fw])

        consecutive_fw_list.append(add_output_files_fw)
        
        parent_links = { consecutive_fw_list[i] : consecutive_fw_list[i+1] \
                        for i in range(len(consecutive_fw_list)-1) }

        return Workflow( consecutive_fw_list, parent_links,
            name="{:s}_production_wf".format(system_name) )
 
def run_minimize(inputs):
    minimize_wfs= ['' for x in range((len(inputs)))] 
    lmp_suffix_template= '-v baseName {baseName:s} -v dataFile {dataFile:s} '
    for i in range((len(inputs))):
        inputs_single=inputs[i]
        minimize_wfs[i]=minimize(inputs_single["system_name"],lmp_suffix_template)
    return FWAction(stored_data={'inputs':inputs}, mod_spec=[{'_set': {'inputs': inputs}}],detours=minimize_wfs )

def run_nvtEquilibrate(inputs):
    nvtEquilibrate_wfs= ['' for x in range((len(inputs)))] 
    lmp_suffix_template='-v baseName {baseName:s} -v dataFile {dataFile:s} '
    for i in range((len(inputs))):
        inputs_single=inputs[i]
        nvtEquilibrate_wfs[i]=nvtEquilibrate(inputs_single["system_name"],lmp_suffix_template)
    return FWAction(stored_data={'inputs':inputs}, mod_spec=[{'_set': {'inputs': inputs}}],detours=nvtEquilibrate_wfs )

def run_nptEquilibrate(inputs):
    nptEquilibrate_wfs= ['' for x in range((len(inputs)))] 
    lmp_suffix_template='-v baseName {baseName:s} -v dataFile {dataFile:s} '
    for i in range((len(inputs))):
        inputs_single=inputs[i]
        nptEquilibrate_wfs[i]=nptEquilibrate(inputs_single["system_name"],lmp_suffix_template)
    return FWAction(stored_data={'inputs':inputs}, mod_spec=[{'_set': {'inputs': inputs}}], detours=nptEquilibrate_wfs )

def run_production(inputs):
    production_wfs= ['' for x in range((len(inputs)))] 
    lmp_suffix_template= '-v baseName {baseName:s} -v dataFile {dataFile:s} '
    for i in range((len(inputs))):
        inputs_single=inputs[i]
        production_wfs[i]=production(inputs_single["system_name"],5000000,lmp_suffix_template)
    return FWAction(detours=production_wfs )


def minimize_cmd(system_name,
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
        
        consecutive_fw_list = []

        get_input_files_ft = GetFilesTask( {
                'identifiers':    ['lmp_header.input', 
                                   'lmp_minimization.input',
                                   'extract_thermo.sh',
                                   system_name+'_psfgen.data' ] ,
                'new_file_names': ['lmp_header.input',
                                   'lmp_minimization.input',
                                   'extract_thermo.sh',
                                   'datafile.lammps']} )
        
        
        get_input_files_fw = Firework( get_input_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_out":  {"lmp_header":'lmp_header.input', 
                                "lmp_minimization":'lmp_minimization.input', 
                                "extract_thermo":'extract_thermo.sh', 
                               "datafile": 'datafile.lammps'},
                "step"   :     "get_files",
                },
            name=(system_name + "get_files") 
        )
        consecutive_fw_list.append(get_input_files_fw)
                
        minimization_ft=CmdTask({'cmd': 'lmp',
                                 'opt':
                                 ['-in lmp_minimization.input',
                                  '-v baseName '+ system_name,
                                  '-v dataFile '+ 'datafile.lammps'
                                 ],
                                 'stdout_file':  'lmp_minimization.out',
                                 'stderr_file':  'lmp_minimization.err',
                                 'use_shell':    True,
                                 'fizzle_bad_rc':True
                                })

        extract_thermo_ft =  CmdTask({'cmd': 'extract_thermo.sh',
                                      'opt':
                                      ['log.lammps',
                                       system_name + '_minimization_thermo.out'],
                                      'stderr_file':'extract_thermo.err',
                                      'stdout_file':'extract_thermo.out',
                                      'fizzle_bad_rc':'false',
                                      'use_shell':'true'
                                     })      
               
        fw = Firework(
            [
                minimization_ft,
                extract_thermo_ft
            ],
            spec={
                "_category":   "nemo_queue_offline",
                "_queueadapter": {"nodes": 8,
                                  "ppn": 20,
                                  "queue": 'nemo_queue_offline',
                                  "walltime": "96:00:00"},
                "_files_in":  {"lmp_header":'lmp_header.input', 
                                "lmp_minimization":'lmp_minimization.input', 
                                "extract_thermo":'extract_thermo.sh', 
                                "datafile": 'datafile.lammps'},
                "_files_out":  {"lmp_minimized":system_name + '.lammps',
                                "thermo_out": system_name + '_minimization_thermo.out'},
                "system_name": system_name,
                "step"     : "minimization"
            },
            name="{:s}_minimzation".format(system_name),
            parents=[get_input_files_fw] )
        
        consecutive_fw_list.append(fw)
        
        add_output_files_ft= AddFilesTask({'compress':True ,
                                           'identifiers': [ system_name + '_minimized.lammps',
                                                           system_name + '_minimization_thermo.out'], 
                                           'paths': [ system_name + '.lammps',
                                                      system_name + '_minimization_thermo.out']})
        
        
        add_output_files_fw = Firework(add_output_files_ft,
            spec = {
                "_category": "nemo_noqueue",
                "system_name" : system_name,
                "_files_in":  { "lmp_minimized":system_name + '.lammps',
                                "thermo_out": system_name + '_minimization_thermo.out' },
                "step"   :     "add_output_files",
                },
            name=(system_name + "add_output_files"),
            parents=[fw]
       )
        consecutive_fw_list.append(add_output_files_fw)
        
        parent_links = { consecutive_fw_list[i] : consecutive_fw_list[i+1] \
                        for i in range(len(consecutive_fw_list)-1) }

        return Workflow( consecutive_fw_list, parent_links,
            name="{:s}_min_wf".format(system_name) )

    
def nvtEquilibrate_cmd(system_name,
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

        consecutive_fw_list = []
        
        get_input_files_ft = GetFilesTask( {
                'identifiers':    ['lmp_header.input', 
                                   'lmp_equilibration_nvt.input', 
                                   'extract_thermo.sh', 
                                   system_name + '_minimized.lammps'
                                  ],
                'new_file_names': ['lmp_header.input', 
                                   'lmp_equilibration_nvt.input',
                                   'extract_thermo.sh',
                                   'datafile.lammps']} )
        
        get_input_files_fw = Firework( get_input_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_out":  {"lmp_header":'lmp_header.input', 
                                "lmp_equilibration":'lmp_equilibration_nvt.input', 
                                "extract_thermo":'extract_thermo.sh', 
                                "datafile": 'datafile.lammps'},
                "step"   :     "get_files",
                },
            name=(system_name + "get_files") 
        )
        consecutive_fw_list.append(get_input_files_fw)
        
        
        lmp_ft=CmdTask({'cmd': 'lmp',
                                 'opt':
                                 ['-in lmp_equilibration_nvt.input',
                                  '-v baseName '+ system_name,
                                  '-v dataFile '+ 'datafile.lammps'
                                  ],
                                 'stdout_file':  'lmp_nvtEquilibration.out',
                                 'stderr_file':  'lmp_nvtEquilibration.err',
                                 'use_shell':    True,
                                 'fizzle_bad_rc':True
                                })

        extract_thermo_ft =  CmdTask({'cmd': 'extract_thermo.sh',
                                      'opt':
                                      [system_name+"_nvtEquilibration.log",
                                       system_name + '_nvtEquilibration_thermo.out'],
                                      'stderr_file':'extract_thermo.err',
                                      'stdout_file':'extract_thermo.out',
                                      'fizzle_bad_rc':'false',
                                      'use_shell':'true'
                                     })      

        fw = Firework(
            [
                lmp_ft,
                extract_thermo_ft
            ],
            spec={
             
                "_category":    "nemo_queue_offline",
                "system_name": system_name,
                "_queueadapter": {"nodes": 8,
                                  "ppn": 20,
                                  "queue": 'nemo_queue_offline',
                                  "walltime": "96:00:00"},
                "_files_in":  { "lmp_header":'lmp_header.input', 
                                "lmp_equilibration":'lmp_equilibration_nvt.input', 
                                "extract_thermo":'extract_thermo.sh', 
                                "datafile": 'datafile.lammps'},
                "_files_out":  {"lmp_nvt": system_name + '_nvtEquilibrated.lammps',
                                "thermo_out": system_name + '_nvtEquilibration_thermo.out'},
                "step"     :   "equilibration_nvt"
            },
            name="{:s}_equilibration_nvt".format(system_name),
            parents=[get_input_files_fw] )
        
        
        consecutive_fw_list.append(fw)
    
        add_output_files_ft= AddFilesTask({'compress':True ,
                                           'identifiers': [ system_name + '_nvtEquilibrated.lammps',
                                                           system_name + '_nvtEquilibration_thermo.out'],
                                           'paths': [ system_name + '_nvtEquilibrated.lammps',
                                                      system_name + '_nvtEquilibration_thermo.out']})
        
        
        add_output_files_fw = Firework(add_output_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_in":  {"lmp_nvt": system_name + '_nvtEquilibrated.lammps',
                               "thermo_out": system_name + '_nvtEquilibration_thermo.out'},
                "step"   :     "add_output_files",
                },
            name=(system_name + "add_output_files"),
            parents=[fw]
       )
        consecutive_fw_list.append(add_output_files_fw)
        
        parent_links = { consecutive_fw_list[i] : consecutive_fw_list[i+1] \
                        for i in range(len(consecutive_fw_list)-1) }

        return Workflow( consecutive_fw_list, parent_links,
            name="{:s}_nvt_wf".format(system_name) )
    


def nptEquilibrate_cmd(system_name,
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
        consecutive_fw_list = []

        get_input_files_ft = GetFilesTask( {
                'identifiers':    ['lmp_header.input', 
                                   'lmp_equilibration_npt.input', 
                                   'extract_thermo.sh',
                                   system_name + '_nvtEquilibrated.lammps'
                                  ],
                'new_file_names': ['lmp_header.input', 
                                   'lmp_equilibration_npt.input', 
                                   'extract_thermo.sh',
                                   'datafile.lammps']} )
        
        get_input_files_fw = Firework( get_input_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_out":  {"lmp_header":'lmp_header.input', 
                                "lmp_equilibration":'lmp_equilibration_npt.input', 
                                "extract_thermo":'extract_thermo.sh',
                                "datafile": 'datafile.lammps'},
                "step"   :     "get_files",
                },
            name=(system_name + "get_files")
                                      
        )
        consecutive_fw_list.append(get_input_files_fw)
        
        
        lmp_ft=CmdTask({'cmd': 'lmp',
                        'opt':
                        ['-in lmp_equilibration_npt.input',
                         '-v baseName '+ system_name,
                         '-v dataFile '+ 'datafile.lammps'],
                        'stdout_file':  'lmp_nptEquilibration.out',
                        'stderr_file':  'lmp_nptEquilibration.err',
                        'use_shell':    True,
                        'fizzle_bad_rc':True
                       })

        extract_thermo_ft =  CmdTask({'cmd': 'extract_thermo.sh',
                                      'opt':
                                      [system_name + '_nptEquilibration.log',
                                       system_name + '_nptEquilibration_thermo.out'],
                                      'stderr_file':'extract_thermo.err',
                                      'stdout_file':'extract_thermo.out',
                                      'fizzle_bad_rc':'false',
                                      'use_shell':'true'
                                     })      
        fw = Firework(
            [
                lmp_ft,
                extract_thermo_ft
            ],
            spec={
                "_category": "nemo_queue_offline",
                "system_name": system_name,
                "_queueadapter": {"nodes": 8,
                                  "ppn": 20,
                                  "queue": 'nemo_queue_offline',
                                  "walltime": "96:00:00"},
                "step"     :   "equilibration_npt",
                "_files_in":  { "lmp_header":'lmp_header.input', 
                                "lmp_equilibration":'lmp_equilibration_npt.input', 
                                "extract_thermo":'extract_thermo.sh',
                                "datafile": 'datafile.lammps'},
                "_files_out": {"lmp_npt":  system_name + '_npt_Equilibrated.lammps',
                               "thermo_out": system_name + '_nptEquilibration_thermo.out'}
            },
            name="{:s}_equilibration_npt".format(system_name),
            parents=[get_input_files_fw])
        
        consecutive_fw_list.append(fw)
        
        add_output_files_ft= AddFilesTask({'compress':True ,
                                           'identifiers': [ system_name + '_nptEquilibrated.lammps',
                                                           system_name + '_nptEquilibration_thermo.out'], 
                                           'paths': [ system_name + '_nptEquilibrated.lammps',
                                                      system_name + '_nptEquilibration_thermo.out']})
        
        add_output_files_fw = Firework(add_output_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_in":  { "lmp_npt": system_name + '_nptEquilibrated.lammps',
                                "thermo_out": system_name + '_nptEquilibration_thermo.out'},
                "step"   :     "add_output_files",
                },
            name=(system_name + "add_output_files"),
            parents=[fw]
       )
        consecutive_fw_list.append(add_output_files_fw)
        
        parent_links = { consecutive_fw_list[i] : consecutive_fw_list[i+1] \
                        for i in range(len(consecutive_fw_list)-1) }

        return Workflow( consecutive_fw_list, parent_links,
            name="{:s}_npt_wf".format(system_name) )

def production_cmd(system_name,
        total_steps      = 5000000, 
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
        consecutive_fw_list = []
        
        get_input_files_ft = GetFilesTask( {'identifiers':    ['lmp_header.input',
                                                               'lmp_production.input',
                                                               'extract_thermo.sh' , 
                                                               system_name + '_nptEquilibrated.lammps'],
                                            'new_file_names': ['lmp_header.input',
                                                               'lmp_production.input',
                                                               'extract_thermo.sh',
                                                               'datafile.lammps']} )
        
        get_input_files_fw = Firework( get_input_files_ft,
            spec = {
                "_category":  "nemo_noqueue",
                "system_name" : system_name,
                "_files_out":  {"lmp_header":'lmp_header.input', 
                                "lmp_production":'lmp_production.input', 
                                "extract_thermo":'extract_thermo.sh',
                                "datafile": 'datafile.lammps'},
                "step"   :     "get_files"
                },
            name=(system_name + "get_files")
                                      
        )
        consecutive_fw_list.append(get_input_files_fw)
                  
        lmp_ft=CmdTask({'cmd': 'lmp',
                                 'opt':
                                 ['-in lmp_production.input',
                                  '-v baseName '+ system_name,
                                  '-v dataFile '+ 'datafile.lammps',
                                  '-v restartFile default.mpiio.restart'
                                  ],
                                 'stdout_file':  'lmp_production.out',
                                 'stderr_file':  'lmp_production.err',
                                 'use_shell':    True,
                                 'fizzle_bad_rc':True
                                })

        extract_thermo_ft =  CmdTask({'cmd': 'extract_thermo.sh',
                                      'opt':
                                      [system_name+'_production.log',
                                       system_name + '_production_thermo.out'],
                                      'stderr_file':'extract_thermo.err',
                                      'stdout_file':'extract_thermo.out',
                                      'fizzle_bad_rc':'false',
                                      'use_shell':'true'
                                     })     

                                     
        fw = Firework(
            [
                lmp_ft,
                extract_thermo_ft

            ],
            spec={
                "_category":     "nemo_queue_offline",
                "system_name":    system_name,
                "_queueadapter": {"nodes": 8,
                                  "ppn": 20,
                                  "queue": 'nemo_queue_offline' ,
                                  "walltime": "96:00:00"},
                "_files_in":  {"lmp_header":'lmp_header.input',
                               "lmp_production":'lmp_production.input',
                               "extract_thermo":'extract_thermo.sh',
                               "datafile": 'datafile.lammps'},
                "_files_out": {"thermo:out": system_name + '_production_thermo.out'},
                "step"     :   "production",
                "total_steps":    total_steps
            },
            name="{:s}_produtcion_{:d}".format(system_name, total_steps) ,
            parents=[get_input_files_fw])

        consecutive_fw_list.append(fw)


        add_output_files_ft= AddFilesTask({'compress': True,
                                           'identifiers': [ system_name + '_production_thermo.out'],
                                           'paths': [ system_name + '_production_thermo.out']})

        add_output_files_fw = Firework(add_output_files_ft,
            spec = {
                "_category": "nemo_noqueue",
                "_files_in":  {"lmp_header":'lmp_header.input', 
                               "lmp_production":'lmp_production.input', 
                               "extract_thermo":'extract_thermo.sh',
                               "datafile": 'datafile.lammps'},
                "_files_out": {"thermo:out": system_name + '_production_thermo.out'},
                "step"     :   "production",
                "total_steps":    total_steps
            },
            name="{:s}_produtcion_{:d}".format(system_name, total_steps) ,
            parents=[get_input_files_fw])
        
        consecutive_fw_list.append(fw)        


        add_output_files_ft= AddFilesTask({'compress': True,
                                           'identifiers': [ system_name + '_production_thermo.out'],
                                           'paths': [ system_name + '_production_thermo.out']})

        add_output_files_fw = Firework(add_output_files_ft,
            spec = {
                "_category": "nemo_noqueue",
                "system_name" : system_name,
                "_files_in": {"thermo_out": system_name + '_production_thermo.out'},
                "step"   : "add_output_files",
                },
            name=(system_name + "add_output_files"),
            parents=[fw])
        consecutive_fw_list.append(add_output_files_fw)

        parent_links = { consecutive_fw_list[i] : consecutive_fw_list[i+1] \
                        for i in range(len(consecutive_fw_list)-1) }

        return Workflow( consecutive_fw_list, parent_links,
            name="{:s}_production_wf".format(system_name) )


def run_minimize_cmd(inputs):
    minimize_wfs= ['' for x in range((len(inputs)))] 
    lmp_suffix_template= '-v baseName {baseName:s} -v dataFile {dataFile:s} '
    for i in range((len(inputs))):
        inputs_single=inputs[i]
        minimize_wfs[i]=minimize_cmd(inputs_single["system_name"],lmp_suffix_template)
    return FWAction(stored_data={'inputs':inputs}, mod_spec=[{'_set': {'inputs': inputs}}],detours=minimize_wfs )

def run_nvtEquilibrate_cmd(inputs):
    nvtEquilibrate_wfs= ['' for x in range((len(inputs)))] 
    lmp_suffix_template='-v baseName {baseName:s} -v dataFile {dataFile:s} '
    for i in range((len(inputs))):
        inputs_single=inputs[i]
        nvtEquilibrate_wfs[i]=nvtEquilibrate_cmd(inputs_single["system_name"],lmp_suffix_template)
    return FWAction(stored_data={'inputs':inputs}, mod_spec=[{'_set': {'inputs': inputs}}],detours=nvtEquilibrate_wfs )

def run_nptEquilibrate_cmd(inputs):
    nptEquilibrate_wfs= ['' for x in range((len(inputs)))] 
    lmp_suffix_template='-v baseName {baseName:s} -v dataFile {dataFile:s} '
    for i in range((len(inputs))):
        inputs_single=inputs[i]
        nptEquilibrate_wfs[i]=nptEquilibrate_cmd(inputs_single["system_name"],lmp_suffix_template)
    return FWAction(stored_data={'inputs':inputs}, mod_spec=[{'_set': {'inputs': inputs}}],detours=nptEquilibrate_wfs )

def run_production_cmd(inputs):
    production_wfs= ['' for x in range((len(inputs)))] 
    lmp_suffix_template= '-v baseName {baseName:s} -v dataFile {dataFile:s} '
    for i in range((len(inputs))):
        inputs_single=inputs[i]
        production_wfs[i]=production_cmd(inputs_single["system_name"],5000000,lmp_suffix_template)
    return FWAction(stored_data={'inputs':inputs}, mod_spec=[{'_set': {'inputs': inputs}}],detours=production_wfs )

