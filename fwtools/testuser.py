from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact
from fireworks.user_objects.firetasks.fileio_tasks import FileTransferTask, FileWriteTask
from fireworks.user_objects.firetasks.script_task import ScriptTask, PyTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask, GetFilesTask, DeleteFilesTask
from fireworks import Firework, LaunchPad, Workflow, FWorker, FireTaskBase, FWAction
import os
import scipy.constants as C
import numpy as np

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
                       "system_name":system_name[i*len(preassembly)+j]}
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
                       "system_name":system_name[i*len(preassembly)+j]}
            inputs[i*len(preassembly)+j]=my_dict
    return inputs
