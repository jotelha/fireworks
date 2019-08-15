""" This module includes dataflow firetask tasks """

__author__ = 'Ivan Kondov'
__email__ = 'ivan.kondov@kit.edu'
__copyright__ = 'Copyright 2016, Karlsruhe Institute of Technology'

import sys
from fireworks import Firework
from fireworks.core.firework import FWAction, FireTaskBase
from fireworks.utilities.fw_serializers import load_object
if sys.version_info[0] > 2:
    basestring = str


class CommandLineTask(FireTaskBase):
    """
    A Firetask to execute external commands in a shell

    Required params:
        - command_spec (dict): a dictionary specification of the command
          (see below for details)

    Optional params:
        - inputs ([str]): list of labels, one for each input argument
        - outputs ([str]): list of labels, one for each output argument
        - chunk_number (int): the serial number of the firetask
          when it is part of a series generated by a ForeachTask
        - env (str): allows to specify an environment possibly defined in the
          worker file. If so, additional environment-related intialization
          and expansion of command aliases are carried out.

    command_spec = {
        'command': [str], # mandatory
        label_1: dict_1, # optional
        label_2: dict_2, # optional
        ...
    }
    The 'command' is a representation of the command as to be used with
    subprocess package. The optional keys label_1, label_2, etc. are
    the actual labels used in the inputs and outputs. The dictionaries dict_1,
    dict_2, etc. have the following schema:
    {
        'binding': {
            prefix: str or None,
            separator: str or None
        },
        'source': {
            'type': 'path' or 'data' or 'identifier'
                     or 'stdin' or 'stdout' or 'stderr' or None,
            'value': str or int or float
        },
        'target': {
            'type': 'path' or 'data' or 'identifier'
                     or 'stdin' or 'stdout' or 'stderr' or None,
            'value': str
        }
    }

    Remarks
    -------

    * If the 'type' in the 'source' field is 'data' the 'value' can be of
    types 'str', 'int' and 'float'.

    * When a *str* is found instead of *dict* for some 'source', for example
    'source': 'string', 'string' is replaced with spec['string'] which must be
    available and of the schema of the 'source'.

    * When a *str* is found instead of *dict* for some label, for example
    label: 'string', 'string' is replaced with spec['string'] which can be a
    dictionary with this schema or a list of such dictionaries.
    """

    _fw_name = 'CommandLineTask'
    required_params = ['command_spec']
    optional_params = ['inputs', 'outputs', 'chunk_number', 'env']

    def run_task(self, fw_spec):
        cmd_spec = self['command_spec']
        ilabels = self.get('inputs')
        olabels = self.get('outputs')
        env = self.get('env')

        if ilabels is None:
            ilabels = []
        else:
            assert isinstance(ilabels, list), '"inputs" must be a list'
        if olabels is None:
            olabels = []
        else:
            assert isinstance(olabels, list), '"outputs" must be a list'

        inputs = []
        outputs = []
        for ios, labels in zip([inputs, outputs], [ilabels, olabels]):
            # cmd_spec: {label: {{binding: {}}, {source: {}}, {target: {}}}}
            for label in labels:
                if isinstance(cmd_spec[label], basestring):
                    inp = []
                    for item in fw_spec[cmd_spec[label]]:
                        if 'source' in item:
                            inp.append(item)
                        else:
                            inp.append({'source': item})
                else:
                    inp = {}
                    for key in ['binding', 'source', 'target']:
                        if key in cmd_spec[label]:
                            item = cmd_spec[label][key]
                            if isinstance(item, basestring):
                                # using ForEachTask, the 'split' list is still
                                # a list. That breaks the functionality here
                                # if used as
                                # { 'split_list': {'source': 'split_list'}
                                # thus try to "remove" encapsulating list
                                if isinstance(fw_spec[item], list):
                                  inp[key] = fw_spec[item][0]
                                else:
                                  inp[key] = fw_spec[item]

                            elif isinstance(item, dict):
                                inp[key] = item
                            else:
                                raise ValueError
                ios.append(inp)

        command = cmd_spec['command']
        command = [ command ] if isinstance(command,basestring) else command
        assert isinstance(command, list)

        # execute environment setup code embedded in worker file
        # TODO: outsource into separate function and provide to other tasks

        # sample settings in worker file:
        # {
        # python: {
        #   init: [
        #     'import sys, os',
        #     'sys.path.insert(0, os.path.join(os.environ["MODULESHOME"], "init"))',
        #     'from env_modules_python import module',
        #     'module("use","/work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles")'
        #   ],
        #   cmd: {
        #     lmp: {
        #       init: 'module("load","lammps/16Mar18-gnu-7.3-openmpi-3.1-colvars-09Feb19")',
        #       prefix: ['mpirun', { 'eval': 'os.environ["MPIRUN_OPTIONS"]' } ]
        #     }
        #   }
        # }

        # check whether worker-specific environment-related setup necessary
        if env:
            if "_fw_env" in fw_spec and env in fw_spec["_fw_env"]:
                # _fw_env : env : init may provide a list of python commans
                # to run, i.e. for module env initialization
                if "init" in fw_spec["_fw_env"][env]:
                    init = fw_spec["_fw_env"][env]["init"]
                    if isinstance(init, basestring): init = [init]
                    assert isinstance(init,list)
                    for cmd in init:
                        exec(cmd)
                else:
                    pass # no particular initialization for this environment

                # check whether there is any machine-specific "expansion" for
                # the command head
                head = command[0]
                if "cmd" in fw_spec["_fw_env"][env] and head in fw_spec["_fw_env"][env]["cmd"]:
                    # same as above, evaluate command-specific initialization code
                    if "init" in fw_spec["_fw_env"][env]["cmd"][head]:
                        init = fw_spec["_fw_env"][env]["cmd"][head]["init"]
                        if isinstance(init, basestring): init = [init]
                        assert isinstance(init,list)
                        for cmd in init:
                            exec(cmd)
                    else:
                        pass # no specific initialization for this command

                    # prepend machine-specific prefixes to command
                    if "prefix" in fw_spec["_fw_env"][env]["cmd"][head]:
                        prefix_list = fw_spec["_fw_env"][env]["cmd"][head]["prefix"]
                        if not isinstance(prefix_list, list):
                            prefix_list = [prefix_list]

                        processed_prefix_list = []
                        for i, prefix in enumerate(prefix_list):
                            processed_prefix = []
                            if isinstance(prefix, dict):
                                # special treatment desired for this prefix
                                if "eval" in prefix:
                                    # evaluate prefix in current context
                                    processed_prefix = eval(prefix["eval"])
                                    try:
                                        processed_prefix = processed_prefix.decode("utf-8")
                                    except AttributeError:
                                        pass
                                    if isinstance(processed_prefix, basestring):
                                        processed_prefix = processed_prefix.split()
                                    else:
                                        raise ValueError(
                                            "Output {} of prefix #{} evaluation not accepted!".format(
                                            processed_prefix, i ) )
                                else:
                                    raise ValueError(
                                        "Formatting {} of prefix #{} not accepted!".format(
                                        prefix, i ) )
                            elif isinstance(prefix,basestring):
                                # prefix is string, not much to do, split & prepend
                                processed_prefix = prefix.split()
                            else:
                                raise ValueError(
                                    "type({}) = {} of prefix #{} not accepted!".format(
                                        prefix, type(prefix), i ) )

                            if not isinstance( processed_prefix, list):
                                processed_prefix = [ processed_prefix ]

                            processed_prefix_list.extend(processed_prefix)


                        command = processed_prefix_list + command # concatenate two lists
                    else:
                        pass # no prefix list to prepend for this command
                else:
                    pass # no command-specific expansion in environment
            else:
                pass # WARNING: environment specified, but not available on worker

        outlist = self.command_line_tool(command, inputs, outputs)

        if len(outlist) > 0:
            if self.get('chunk_number') is not None:
                mod_spec = []
                if len(olabels) > 1:
                    assert len(olabels) == len(outlist)
                    for olab, out in zip(olabels, outlist):
                        for item in out:
                            mod_spec.append({'_push': {olab: item}})
                else:
                    for out in outlist:
                        mod_spec.append({'_push': {olabels[0]: out}})
                return FWAction(mod_spec=mod_spec)
            else:
                output_dict = {}
                for olab, out in zip(olabels, outlist):
                    output_dict[olab] = out
                return FWAction(update_spec=output_dict)
        else:
            return FWAction()

    @staticmethod
    def command_line_tool(command, inputs=None, outputs=None):
        """
        This function composes and executes a command from provided
        specifications.

        Required parameters:
            - command ([str]): the command as to be passed to subprocess.Popen

        Optional parameters:
            - inputs ([dict, [dict]]): list of the specifications for inputs;
              multiple inputs may be passed in one list of dictionaries
            - outputs ([dict]): list of the specifications for outputs

        Returns:
            - list of target dictionaries for each output:
                'target': {
                    'type': 'path' or 'data' or 'identifier'
                             or 'stdin' or 'stdout' or 'stderr' or None
                    'value': str
                }
              If outputs is None then an empty list is returned.
        """
        import os
        import uuid
        from subprocess import Popen, PIPE
        from shutil import copyfile

        def set_binding(arg):
            argstr = ''
            if 'binding' in arg:
                if 'prefix' in arg['binding']:
                    argstr += arg['binding']['prefix']
                if 'separator' in arg['binding']:
                    argstr += arg['binding']['separator']
            return argstr

        arglist = command
        stdin = None
        stdout = None
        stderr = PIPE
        stdininp = None
        if inputs is not None:
            for inp in inputs:
                argl = inp if isinstance(inp, list) else [inp]
                for arg in argl:
                    argstr = set_binding(arg)
                    assert 'source' in arg, 'input has no key "source"'
                    assert (arg['source']['type'] is not None
                            and arg['source']['value'] is not None)
                    if 'target' in arg:
                        assert arg['target'] is not None
                        assert arg['target']['type'] == 'stdin'
                        if arg['source']['type'] == 'path':
                            stdin = open(arg['source']['value'], 'r')
                        elif arg['source']['type'] == 'data':
                            stdin = PIPE
                            stdininp = str(arg['source']['value']).encode()
                        else:
                            # filepad
                            raise NotImplementedError()
                    else:
                        if arg['source']['type'] == 'path':
                            argstr += arg['source']['value']
                        elif arg['source']['type'] == 'data':
                            argstr += str(arg['source']['value'])
                        else:
                            # filepad
                            raise NotImplementedError()
                    if len(argstr) > 0:
                        arglist.append(argstr)

        if outputs is not None:
            for arg in outputs:
                if isinstance(arg, list):
                    arg = arg[0]
                argstr = set_binding(arg)
                assert 'target' in arg
                assert arg['target'] is not None
                if arg['target']['type'] == 'path':
                    assert 'value' in arg['target']
                    assert len(arg['target']['value']) > 0
                    path = os.path.abspath( arg['target']['value'] )

                    if os.path.isdir(path):
                        path = os.path.join(path, str(uuid.uuid4()))

                    arg['target']['value'] = path
                    if 'source' in arg:
                        assert arg['source'] is not None
                        assert 'type' in arg['source']
                        if arg['source']['type'] == 'stdout':
                            stdout = open(path, 'w')
                        elif arg['source']['type'] == 'stderr':
                            stderr = open(path, 'w')
                        elif arg['source']['type'] == 'path':
                            pass
                        else:
                            argstr += path
                    else:
                        argstr += path
                elif arg['target']['type'] == 'data':
                    stdout = PIPE
                else:
                    # filepad
                    raise NotImplementedError()
                if len(argstr) > 0:
                    arglist.append(argstr)

        proc = Popen(arglist, stdin=stdin, stderr=stderr, stdout=stdout)
        res = proc.communicate(input=stdininp)
        if proc.returncode != 0:
            err = res[1] if len(res) > 1 else ''
            raise RuntimeError(err) # TODO: option to not fizzle

        retlist = []
        if outputs is not None:
            for output in outputs:
                if ('source' in output
                        and output['source']['type'] == 'path'):
                    # above fails if source is "list"
                    copyfile(
                        output['source']['value'],
                        output['target']['value']
                    )
                if output['target']['type'] == 'data':
                    output['target']['value'] = res[0].decode().strip()
                retlist.append(output['target'])

        return retlist


class ForeachTask(FireTaskBase):
    """
    This firetask branches the workflow creating parallel fireworks
    using FWAction: one firework for each element or each chunk from the
    *split* list. Each firework in this generated list contains the firetask
    specified in the *task* dictionary. If the number of chunks is specified
    the *split* list will be divided into this number of chunks and each
    chunk will be processed by one of the generated child fireworks.

    Required params:
        - task (dict): a dictionary version of the firetask
        - split (str or [str]): label  an input list or a list of such;
          they must be available both in
          the *inputs* list of the specified task and in the spec.

    Optional params:
        - number of chunks (int): if provided the *split* input list will be
          divided into this number of sublists and each will be processed by
          a separate child firework
        - chunk index spec (str): if provided, chunk index is
          stored in this _fw_spec field
    """
    _fw_name = 'ForeachTask'
    required_params = ['task', 'split']
    optional_params = ['number of chunks']

    def run_task(self, fw_spec):
        assert isinstance(self['split'], (basestring,list)), self['split']
        split_list = self['split']
        if isinstance( split_list, basestring): split_list = [split_list]

        reflen = 0
        for split in split_list:
            assert isinstance(fw_spec[split], list)
            if isinstance(self['task']['inputs'], list):
                assert split in self['task']['inputs']
            else: # only one inputs entry , str
                assert split == self['task']['inputs']

            split_field = fw_spec[split]
            lensplit = len(split_field)

            # update reflen on first iteration
            if reflen == 0:
                assert lensplit != 0, ('input to split is empty:', split)
                reflen = lensplit
                nchunks = self.get('number of chunks')
                if not nchunks:
                    nchunks = lensplit
                chunklen = lensplit // nchunks
                if lensplit % nchunks > 0:
                    chunklen = chunklen + 1

                chunks = [ { split: split_field[i:i+chunklen] } for i in range(0, lensplit, chunklen)]
            else:
                assert lensplit == reflen, ('input lists not of equal length:', split)
                for i in range(0, lensplit, chunklen):
                    chunks[i//chunklen].update( { split: split_field[i:i+chunklen] } )

        fireworks = []
        chunk_index_spec = self.get('chunk index spec')

        for index, chunk in enumerate(chunks):
            spec = fw_spec.copy()
            for split in split_list:
                spec[split] = chunk[split]
            task = load_object(self['task'])
            task['chunk_number'] = index
            if chunk_index_spec and isinstance(chunk_index_spec, basestring):
                spec[chunk_index_spec] = index
            name = self._fw_name + ' ' + str(index)
            fireworks.append(Firework(task, spec=spec, name=name))
        return FWAction(detours=fireworks)


class JoinDictTask(FireTaskBase):
    """ combines specified spec fields into a dictionary """
    _fw_name = 'JoinDictTask'
    required_params = ['inputs', 'output']
    optional_params = ['rename']

    def run_task(self, fw_spec):
        assert isinstance(self['output'], basestring)
        assert isinstance(self['inputs'], list)

        if self['output'] not in fw_spec:
            output = {}
        else:
            assert isinstance(fw_spec[self['output']], dict)
            output = fw_spec[self['output']]

        if self.get('rename'):
            assert isinstance(self.get('rename'), dict)
            rename = self.get('rename')
        else:
            rename = {}
        for item in self['inputs']:
            if item in rename:
                output[self['rename'][item]] = fw_spec[item]
            else:
                output[item] = fw_spec[item]

        return FWAction(update_spec={self['output']: output})


class JoinListTask(FireTaskBase):
    """ combines specified spec fields into a list. """
    _fw_name = 'JoinListTask'
    required_params = ['inputs', 'output']

    def run_task(self, fw_spec):
        assert isinstance(self['output'], basestring)
        assert isinstance(self['inputs'], list)
        if self['output'] not in fw_spec:
            output = []
        else:
            assert isinstance(fw_spec[self['output']], list)
            output = fw_spec[self['output']]

        for item in self['inputs']:
            output.append(fw_spec[item])

        return FWAction(update_spec={self['output']: output})


class ImportDataTask(FireTaskBase):
    """
    Update the spec with data from file in a nested dictionary at a position
    specified by a mapstring = maplist[0]/maplist[1]/...
    i.e. spec[maplist[0]][maplist[1]]... = data
    """

    _fw_name = 'ImportDataTask'
    required_params = ['filename', 'mapstring']
    optional_params = []

    def run_task(self, fw_spec):
        from functools import reduce
        import operator
        import json
        import ruamel.yaml as yaml

        filename = self['filename']
        mapstring = self['mapstring']
        assert isinstance(filename, basestring)
        assert isinstance(mapstring, basestring)
        maplist = mapstring.split('/')

        fmt = filename.split('.')[-1]
        assert fmt in ['json', 'yaml']
        with open(filename, 'r') as inp:
            data = json.load(inp) if fmt == 'json' else yaml.safe_load(inp)

        leaf = reduce(operator.getitem, maplist[:-1], fw_spec)
        if isinstance(data, dict):
            if maplist[-1] not in leaf:
                leaf[maplist[-1]] = data
            else:
                leaf[maplist[-1]].update(data)
        else:
            leaf[maplist[-1]] = data

        return FWAction(update_spec={maplist[0]: fw_spec[maplist[0]]})
