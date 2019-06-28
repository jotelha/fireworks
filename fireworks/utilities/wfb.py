""" Workflow Builder, facilitates workflow constructionm """

__author__ = 'Johannes Hörmann'
__email__ = 'johannes.hoermann@imtek.uni-freiburg.de'
__copyright__ = 'Copyright 2019, University of Freiburg'

# IO, formats:
import logging, os.path
from tabulate import tabulate
from pprint import pprint
import yaml
from jinja2 import Template, Environment, FileSystemLoader
from jinja2 import select_autoescape, meta
from fireworks import Firework, Workflow

# graphs and datastructures
import igraph, itertools
import numpy as np
from collections.abc import Iterable

# custom jinja2 filters
import time
def datetime(value,format='%Y-%m-%d-%H:%M'):
    if value == 'now':
        return time.strftime(format)
    else:
        return value.strftime(format)

class WorkflowBuilder:
    std_context         = {}
    persistent_contexts = {}
    transient_contexts  = {}
    dependencies        = {}
    metadata            = {}

    template_dir = 'templates'
    build_dir = 'build'

    g = igraph.Graph(directed=True)
    h = igraph.Graph(directed=True)

    t = None # degenerate tree

    topological_order = None

    root = None

    #  template engine
    env = None

    logger = None

    # plot settings
    plt_layout = None
    plt_label_font_size = 10
    plt_bbox = (800, 600)
    plt_margin = 100

    @property
    def topological_order(self):
        return list(reversed(self.postorder()))

    @property
    def topological_position(self):
        """A list assigning its position in a topological ordering to a vertex"""
        return { v: i for i, v in enumerate(self.topological_order) }

    @property
    def maximum_distance(self):
        """A list assigning the largest possible path length from root to vertex"""
        return [(-1)*int(d) for d in self.g.shortest_paths(self.root, weights=-np.ones(len(self.g.es)))[0] ]

    @property
    def names(self):
        """Returns list of vertex names for list of indices"""
        return self.g.vs["name"]

    def positive_fw_id_generator(self):
        i = 0
        while True:
            yield 10*(i+1)
            i = i +1

    def __init__(self, system_file = None, loglevel = None):
        self.logger = logging.getLogger(__name__)
        if loglevel is not None:
            self.logger.setLevel(loglevel)
        if system_file is not None:
            self.read_system_file(system_file)
            self.build_graph()

    def read_system_file(self, system_file):
        assert type(system_file) is str

        with open(system_file) as stream:
            system = yaml.safe_load(stream)

        self.logger.info("File '{:s} contains the sections {}.".format( system_file, list(system.keys())) )
        assert "std"          in system
        assert "transient"    in system
        assert "persistent"   in system
        assert "dependencies" in system
        assert "name"         in system
        assert "metadata"     in system

        self.std_context         = system['std']
        self.persistent_contexts = system['persistent']
        self.transient_contexts  = system['transient']
        self.dependencies        = system['dependencies']
        self.name                = system["name"]
        self.metadata            = system["metadata"]

    def initialize_template_engine(self):
        self.env = Environment(
          loader=FileSystemLoader(self.template_dir),
          autoescape = False)
        #  autoescape=select_autoescape(['yaml']))
        # register filters:
        self.env.filters['datetime'] = datetime

    def render_template(self,template_name,outfile_name,context):
        template = self.env.get_template(template_name)
        output = template.render(context)
        outfile_name = os.path.join(self.build_dir,outfile_name)
        with open(outfile_name, 'w') as of:
            of.write(output)
        return

    def fill_templates(self):
        dep = {}
        for v in self.topological_order:
            template =  self.g.vs[v]["template"]
            context =   self.g.vs[v]["transient"]
            outfile =   self.g.vs[v]["name"] # all anmes should be unique!

            try:
              self.render_template(template, outfile, context)
            except:
              self.logger.exception("Error rendering template '{:s}'!".format(template))
              raise

            dep[outfile] = [ self.g.vs[c]["name"] for c in self.g.neighbors(v,mode=igraph.OUT) ]

        with open(os.path.join(self.build_dir,"dependencies.yaml"), 'w') as f:
            yaml.dump(dep, f, default_flow_style=False)

        with open(os.path.join(self.build_dir,"metadata.yaml"), 'w') as f:
            yaml.dump(self.metadata, f, default_flow_style=False)

        with open(os.path.join(self.build_dir,"name.yaml"), 'w') as f:
            yaml.dump(self.name, f, default_flow_style=False)

        return

    def compile_workflow(self, outfile='wf.yaml', from_files=True):
        # negative fw ids:
        with open(os.path.join(self.build_dir,'dependencies.yaml'),'r') as stream:
            dependencies = yaml.safe_load(stream)
        with open(os.path.join(self.build_dir,'metadata.yaml'),'r') as stream:
            metadata = yaml.safe_load(stream)
        with open(os.path.join(self.build_dir,'name.yaml'),'r') as stream:
            wf_name = yaml.safe_load(stream)

        fws_set = set()
        for k,v in dependencies.items():
            fws_set.update([k,*v])
        self.logger.info("Set of Fireworks:{}.".format( sorted(fws_set) ) )

        fws_id = { fw: (-10*(i+1)) for i, fw in enumerate( sorted(fws_set) ) }

        fws_dict = {}
        for fw in sorted(fws_set):
            self.logger.info("Loading files '{}'.".format( fw ) )
            fws_dict[fw] = Firework.from_file(os.path.join(
                self.build_dir, fw), f_format='yaml')

        for name, fw in fws_dict.items():
            fw.fw_id = fws_id[name]
            self.logger.info("fw_id {: 4d}: {:s}".format(fw.fw_id, name))

        fws_list = list(fws_dict.values())

        links = {
            fws_dict[parent]: [
                fws_dict[child] for child in children ] for parent, children in dependencies.items() }

        wf = Workflow( fws_list, links, name=wf_name, metadata=metadata )
        wf.to_file(os.path.join(self.build_dir,outfile),f_format="yaml")

        return wf


    def find_undefined_variables(self):
        template_variables = { 'all' : set() }
        for template_name in self.env.list_templates():
            self.logger.info("Loading template {:s}.".format( template_name ) )
            template_source = self.env.loader.get_source(self.env,template_name)[0]
            try:
                parsed_content = self.env.parse(template_source)
            except:
                self.logger.exception("Failed parsing template '{:s}'".format(template_name))
                raise
            template_variables[template_name] = meta.find_undeclared_variables(parsed_content)
            template_variables['all'] = template_variables['all'] | template_variables[template_name]
        return template_variables

    def variable_overview(self, variables):
        lines = [ [ '', *variables['all'] ] ]
        for t,tv in variables.items():
            if t is not 'all':
                line = [ t ]
                line.extend( [ 'x' if v in tv else '' for v in variables['all'] ] )
                lines.append(line)
        return tabulate(lines,tablefmt='fancy_grid')

    def show_undefined_variables(self):
        """Show a table of all undefined variables in templates"""
        return self.variable_overview( self.find_undefined_variables() )

    def build_graph(self):
        fws_set = set()
        for k,vv in self.dependencies.items():
            v = [ list(v.keys())[0] if type(v) is dict else v for v in vv ]
            fws_set.update([k,*v])

        self.g = igraph.Graph(directed=True)
        self.g.add_vertices(sorted(list(fws_set)))
        self.logger.info("Graph contains {:d} nodes: {}.".format( len(self.g.vs), self.g.vs["name"] ) )

        #edges = [ (parent, child) for parent, children in self.dependencies.items() for child in children ]
        for parent, children in self.dependencies.items():
            for child in children:
                if type(child) is dict:
                    edge_type = list(child.values())[0] # expect only one entry
                    child     = list(child.keys())[0]
                else:
                    edge_type = 0 # default
                self.g.add_edge(parent,child,type=edge_type)
        self.logger.info("Graph contains {:d} edges: {}.".format( len(self.g.es),
            [ (self.g.vs[ self.g.es[e].source ]["name"], self.g.vs[ self.g.es[e].target ]["name"] ) for e in self.g.es.indices ] ) )

        #self.g.vs["instance"] = [[]]*len(self.g.vs)   # list of instances
        self.g.vs["persistent"] = [[{}]]*len(self.g.vs) # list of persistent contexts
        self.g.vs["transient"] = [[{}]]*len(self.g.vs)  # list of transient contextes 
        for v in self.g.vs:
            if v["name"] in self.persistent_contexts:
                v["persistent"] = self.persistent_contexts[ v["name"] ]
            if v["name"] in self.transient_contexts:
                v["transient"] = self.transient_contexts[ v["name"] ]
                
        self.g.vs["template"] = self.g.vs["name"]  # this must not be touched later 

        self.update()

    def update(self):
        if not self.g.is_dag():
            self.logger.exception("Graph is not DAG!")
            raise ValueError
            
        root = np.where(np.equal(self.g.degree(mode=igraph.IN),0))[0]
        if len(root) == 0:
            self.logger.exception("Graph has no root!")
            raise ValueError
        elif len(root) > 1:
            self.logger.exception("Graph root not unique: {}!".format( [ self.g.vs[v]["name"] for v in root ]))
            raise ValueError

        self.root = int(root[0])
        self.logger.info("Identified root {}.".format( self.g.vs[self.root]["name"] ) )

        self.g.vs["order"] = self.topological_order
        self.g.vs["dist"] = self.maximum_distance

        # for plotting purposes:
        # self.plt_layout = self.g.layout_auto() 
        try:
            #self.plt_layout = self.g.layout_reingold_tilford(mode="out", root=[self.root]) # tree-like
            self.g.layout_fruchterman_reingold() # better than Kamanda-Kawai
        except:
            #self.plt_layout = self.g.layout_kamada_kawai()
            self.plt_layout = self.g.layout_sugiyama()

        self.g.vs["label"] = self.g.vs["name"]
        self.g.vs["label_size"] = self.plt_label_font_size


    def subgraph_at(self, v, x=set()):
        """Returns indices of all vertices in the subgraph beginning at node v, not following any edges leading to vertices in x"""
        if not isinstance(x, Iterable):
            x = set([x])
            
        visited = set()
        def dfs(v):
            visited.add(v)
            for child in self.g.neighbors(v,mode=igraph.OUT):
                if (child not in x) and (child not in visited):
                    dfs(child)
        dfs(v)
        return visited
    
    def subgraphs_below(self, v):
        """Returns list of list of indices of all vertices in the subgraphs beginning below node v"""
        forks = {}
        for eid in self.g.incident(v,mode=igraph.OUT):
            e = self.g.es[eid]
            s = self.subgraph_at(e.target)
            if e["type"] in forks:
                forks[ e["type"] ].update(s)
            else:
                forks[ e["type"] ] = set(s)
            
        return list(forks.values())
    
    def subgraph_except(self, x):
        """Returns indices of all vertices in the subgraph not following any path to any vertex in x"""
        visited = set()
        def dfs(v):
            visited.add(v)
            for child in self.g.neighbors(v,mode=igraph.OUT):
                if (child is not x) and (child not in visited):
                    dfs(child)
        dfs(self.root)
        return visited
        
        # visited has now all vertices of subgraph
        self.logger.info("Selected subgraph with nodes: {}.".format( internal ) )
        
    def duplicate_subset(self,sub,env,suffix='*'):
        """Duplicates subset within environment and returns old vid -> new vid dict mapping"""
        adjacent = set()
        
        def new_name(v):
            return (self.g.vs[v]["name"]+str(suffix))
        
        def duplicate_vertex(v):
            nam = new_name(v)
            self.h.add_vertex( nam )
            w = self.h.vs.find(name=nam).index
            self.logger.info("Added vertex {:d} - {:s}".format( w, nam ) )
            for attribute in self.g.vs.attribute_names():
                if attribute is not "name":
                    self.h.vs[w][attribute] = self.g.vs[v][attribute]
            self.h.vs[w]["label"] = self.h.vs[w]["name"]
            return w
        
        def duplicate_edge(e,s,t):
            self.h.add_edge(s,t)
            f = self.h.get_eid(s,t)
            self.logger.info("Added edge {:d} - ({:d}-{:d})".format( f, s, t ) )
            for attribute in self.g.es.attribute_names():
                self.h.es[f][attribute] = self.g.es[e][attribute]
            return f
        
        def edges_adjecent_to_subset(s):
            edges = set()
            for v in s:
                edges.update( self.g.incident(v,mode=igraph.ALL) )
            return edges
        
        dup = [ duplicate_vertex(v) for v in sub ]
        sub_to_dup = { s: d for s,d in zip(sub,dup) }
        
        edges = edges_adjecent_to_subset(sub)
        
        for e in edges:
            (source,target) = self.g.es[e].tuple
            
            if (source in sub) and (target in sub):
                duplicate_edge(e,sub_to_dup[source],sub_to_dup[target])
            elif (source in sub) and (target in env):
                duplicate_edge(e,sub_to_dup[source],target)
            elif (source in env) and (target in sub):        
                duplicate_edge(e,source,sub_to_dup[target])
                    
        return sub_to_dup
    
    def descend(self):
        """Descends the graph in topological order and creates forks where demanded
        
        Currently only supports bifurcations"""
        
        self.g.vs["visited"] = False
        
        # the topological order might change as new vertices are added during descend
        # introduced marker "visited" for this purpose
        suffix=''
        while True:
            vs = self.topological_order
            self.logger.info("Current topological order: {}".format( vs ) )

            for v in vs:
                #v = self.topological_order[i]
                if self.g.vs[v]["visited"] is True:
                    self.logger.info("{:d}: {} has been visited before, move on.".format(v, self.g.vs[v]["name"] ) )
                    continue

                self.logger.info("{:d}: {}.".format(v, self.g.vs[v]["name"] ) )

                self.g.vs[v]["visited"] = True
                #get all forks if vertex marked accordingly
                forks = self.subgraphs_below(v)  

                # so far only supports bifurcation
                if len(forks) > 1:
                    self.logger.info("{:d} forks at v: {}.".format( len(forks), forks ) )

                    fl = forks.pop()
                    self.h = self.g.copy()
                    # TODO: implement support for > 2 fork
                    # use first fork as reference
                    # process forks pairwise:

                    fr = forks.pop()

                    self.logger.info("Lef and right fork: {}.".format( fl, fr ) )
                    overlap = fl & fr
                    self.logger.info("Overlap: {}.".format( overlap ) )


                    #env = set(self.topological_order) # a set of all nodes
                    env = set(self.subgraph_at(self.root)) # a set of all nodes
                    self.logger.info("Current set of vertices: {}.".format( env ) )
                    
                    suffix = suffix + '*'

                    lenv = (env - fr ) | fl
                    self.logger.info("Environment of left fork: {}.".format( lenv ) )
                    self.duplicate_subset(overlap,lenv,suffix)

                    suffix = suffix + '*'

                    renv = (env - fl ) | fr
                    self.logger.info("Environment of right fork: {}.".format( lenv ) )

                    self.duplicate_subset(overlap,renv,suffix)

                    #to_delete.update(overlap)
                    self.h.delete_vertices(overlap)
                    self.g = self.h
                    self.update() # root index might change
                    
                    # go up one level to while loop and rebuild topological order
                    # after modification of graph
                    self.logger.info("Forked and modified graph at node {}, rebuilding topological order.".format( v ) )
                    break
            else: # only if inner loop did not break:
                self.logger.info("All processed at node {}, finished.".format( v ) )
                break # all vertices v have been visited, also in extended graph
            self.logger.info("Completed processing node {}, descending.".format( v ) )
        return    
            
    def maximum_spanning_tree(self):
        """Constructs maximum spanning tree by negation of weights"""
        #minimum_spanning_tree = self.g.spanning_tree()
        negated_weights = -np.ones(len(self.g.es))
        return self.g.spanning_tree(weights=negated_weights)
    
    def bfs(self):
        return self.g.bfs(self.root, mode = igraph.OUT)
        
    def bfs_advanced(self):
        bfs = self.g.bfsiter(self.root, mode = igraph.OUT, advanced = True)
        for (v, depth, p) in bfs:
            if p is not None:
                self.logger.info("{:03d}: {:s} - depth {:0d},  {:03d}: {:s}".format(v.index, v["name"], depth, p.index, p["name"] ))
            else:
                self.logger.info("{:03d}: {:s} - depth {:0d}, no ".format(v.index, v["name"], depth))
                
    def tree(self):
        """Yields tree with possibly degenerate vertices"""
        t, m = self.g.unfold_tree(self.root, mode=igraph.OUT)
        for v in t.vs:
            v["name"] = self.g.vs[ m[v.index] ]["name"]
        t.vs["label"] = t.vs["name"]
        t.vs["label_size"] = self.plt_label_font_size
        self.t = t
        return t
    
    def postorder(self):
        """Visit all vertices in post-order beginning at root"""
        visited = set()
        order = []
        def dfs(v):
            visited.add(v)
            for child in self.g.neighbors(v,mode=igraph.OUT):
                if child not in visited:
                    dfs(child)
            order.append(v)
        dfs(self.root)
        return order
    
    def select_closest_parent(self, v):
        """Returns parent closest above v, i.e. farthest from root"""
        parents = self.g.neighbors(v,mode=igraph.IN)
        selected_parent = None
        if len(parents) > 0:
            parent_distances = [ self.maximum_distance[p] for p in parents ]

            self.logger.info("{:d}: {:s}, dist. {:d}, top.pos. {:d} has {:d} parents with distances {}:".format(
               v, self.g.vs[v]["name"], self.g.vs[v]["dist"], self.g.vs[v]["order"], len(parents), parent_distances))
            for p in parents:
                self.logger.info("  {:d}: {:s}, dist. {:d}, top. pos. {:d}".format(
                    p, self.g.vs[p]["name"], self.g.vs[p]["dist"], self.g.vs[p]["order"]))

            # select indices of closest parents
            closest_parents = [ parents[p] for p in np.where( np.equal( np.max(parent_distances), parent_distances ) )[0] ]

            self.logger.info("    Maximum distance {:d} at parent {}".format(np.max(parent_distances),closest_parents) )

            if len(closest_parents) > 1:
                self.logger.warn("  {:d} parents are equally closest to child: Choice of context not unique.".format(len(closest_parents)))
                for p in closest_parents:
                    self.logger.warn("    {:d}: {:s}, dist. {:d}, top. pos. {:d}".format(
                        p, self.g.vs[p]["name"], self.g.vs[p]["dist"], self.g.vs[p]["order"]))
                self.logger.warn("  Choice of context not unique.")

            selected_parent = closest_parents[0]
            self.logger.info("  {:d}: {:s}, dist. {:d}, top. pos. {:d} selected as immediate parent".format(
                selected_parent, self.g.vs[selected_parent]["name"], self.g.vs[selected_parent]["dist"], self.g.vs[selected_parent]["order"]))
        else:
            self.logger.info("{:d}: {:s}, dist. {:d}, top.pos. {:d} has no parents.".format(v, self.g.vs[v]["name"], self.g.vs[v]["dist"], self.g.vs[v]["order"]))
        return selected_parent

    def build_degenerate_graph(self):
        # traverse in topological order to assure all parents have been processed before their children
        fwId = self.positive_fw_id_generator()

        self.g.vs["visited"] = False

        # self.h = igraph.Graph(directed=True)

        while True:
            vs = self.topological_order
            self.logger.info("Current topological order: {}".format( vs ) )

            env = self.subgraph_at(self.root) # the whole graph

            for v in vs:
                #v = self.topological_order[i]
                if self.g.vs[v]["visited"] is False:
                    break
                self.logger.info("{:d}: {} has been visited before, move on.".format(v, self.g.vs[v]["name"] ) )
            else:
                break # all v visted

            self.logger.info("{:d}: {}.".format(v, self.g.vs[v]["name"] ) )
            self.logger.info("Iterating node {:d}: {:s}".format(v, self.g.vs[v]["name"] ))

            self.g.vs[v]["visited"] = True
            #get all forks if vertex marked accordingly
            #forks = self.subgraphs_below(v)
            sub = self.subgraph_at(v) # all vertices below v

            fw_class_name = self.g.vs[v]["template"]
            if fw_class_name not in self.env.list_templates():
                    raise ValueError("No template '{:s}' exists!".format(fw_class_name))
            # self.logger.warn("{}{}:".format(' '*depth, ))

            # children = g.neighbors(v,mode=igraph.OUT)
            # select  with largest distance to root as immediate predecessor
            # this is not well defined for parents on the same level
            # parents = self.g.neighbors(v,mode=igraph.IN)

            parent_id = self.select_closest_parent(v)
            if parent_id is not None:
                # parent_instances = self.h.vs.select(instanceOf=parent_id)
                parent_context = self.h.vs[parent_id]["persistent"] # inherit context
                parent_degeneracy = self.h.vs[parent_id]["degeneracy"]
                parent_name = self.h.vs[parent_id]["name"]
            else: # root node
                parent_context = self.std_context
                parent_degeneracy = 1
                parent_name = None

            persistent_context_updates = self.g.vs[v]["persistent"]
            transient_context_updates = self.g.vs[v]["transient"]

            degeneracy = parent_degeneracy*len(persistent_context_updates)*len(transient_context_updates)

            self.logger.info("Degeneracy (persistent*transient): {:d}*{:d} = {:d}".format(
                len(persistent_context_updates),len(transient_context_updates),degeneracy))

            self.logger.debug("  Parent name: {}".format(parent_name))
            self.logger.debug("  Parent context: {}".format(parent_context))
            self.logger.debug("  Child persistent context updates {}".format(persistent_context_updates))
            self.logger.debug("  Child transient context updates {}".format(transient_context_updates))
            # take all  instances and first apply ...

            # return zip(parent_names, parent_contexts)
            # if degeneracy > 1
                # for (parent_name, parent_context) in zip(parent_names, parent_contexts):

            self.logger.info("  Iterating parent {}:".format(parent_name))

            self.h = self.g.copy()
            for i, persistent_context_update in enumerate(persistent_context_updates):
                self.logger.info("    Iterating persitent content update {:d}:".format(i))
                # ... persitent changes ...
                persistent_context = parent_context.copy()
                persistent_context.update( persistent_context_update )
                self.logger.debug("      Current persistent context: {}".format(persistent_context))

                for j, transient_context_update in enumerate(transient_context_updates):
                    self.logger.info("      Iterating transient content update {:d}:".format(j))
                    # then apply transient changes ...
                    transient_context = persistent_context.copy()
                    transient_context.update( transient_context_update )

                    self.logger.debug("        Current transient context: {}".format(transient_context))

                    fw_id = next(fwId) # get unique id

                    suffix = "_{:06d}".format(fw_id)
                    child_name = "{:s}{:s}".format(fw_class_name,suffix)
                    sub_to_dup = self.duplicate_subset(sub, env, suffix)

                    # update properties of new subgraph root
                    self.h.vs[ sub_to_dup[v] ]["name"] = child_name
                    self.h.vs[ sub_to_dup[v] ]["persistent"] = persistent_context
                    self.h.vs[ sub_to_dup[v] ]["transient"]  = transient_context
                    self.h.vs[ sub_to_dup[v] ]["degeneracy"] = degeneracy
                    self.h.vs[ sub_to_dup[v] ]["fw_id"]      = fw_id
                    # self.h.add_vertex(
                    #    child_name,
                    #    instanceOf=v,
                    #    fw_id=fw_id,
                    #    persistent=persistent_context,
                    #    transient=transient_context)
                    # child_id = self.h.vs.find(name=child_name).index

                    self.logger.info("          Instance {:d} - {:d} - {:s} of class {:d} - {:s} created.".format(
                        sub_to_dup[v],
                        fw_id,
                        child_name,
                        v,
                        self.g.vs[v]["template"] ) )
                    #if parent_name is not None:
                    #    self.h.add_edge(parent_name, child_name)
                    #    parent_id = h.vs.find(name=parent_name).index
                    #    self.logger.info("          Attached to instance {:d} - {:d} - {:s} of class {:d}: {:s}.".format(
                    #        parent_id,
                    #        h.vs[parent_id]["fw_id"],
                    #        h.vs[parent_id]["name"],
                    #        h.vs[parent_id]["instanceOf"],
                    #        self.g.vs[ h.vs[parent_id]["instanceOf"] ]["name"]))
                    #to_delete.update(overlap)

                    # go up one level to while loop and rebuild topological order
                    # after modification of graph
                    self.logger.info("      Completed transient content update {:d}:".format(j))

                self.logger.info("    Completed persitent content update {:d}:".format(i))
            #self.logger.info("  Completed parent {}:".format(parent_name))

            self.h.delete_vertices(sub)
            self.g = self.h
            self.update() # root index might have changed

            self.logger.info("      Modified graph at node {}, rebuilding topological order.".format( v ) )
            self.logger.info("Completed node {:d}: {:s}".format(v, self.g.vs[v]["name"] ) )

        self.logger.info("All processed at node {}, finished.".format( v ) )

        return

    def plot(self, g = None):
        if g is None:
            g = self.g
        return igraph.plot(g, layout = self.plt_layout, bbox = self.plt_bbox, margin = self.plt_margin)

    def show_attributes(self, vs=None, exclude=[
        "name", "persistent", "transient", "label", "label_size", "visited"] ):
        if vs is None:
            vs = self.g.vs
        else:
            vs = self.g.vs.select(vs)

        return tabulate(
            [
                [ '', "name", *[a for a in vs.attribute_names() if a not in exclude] ],
                *list(
                    zip(
                        [ i for i in range(len(vs)) ],
                        vs["name"],
                        *[ vs[a] for a in vs.attribute_names() if a not in exclude ] ) ) ] )

def build_wf(system_infile = 'system.yaml', build_dir = 'build', template_dir ) 'templates'
    """Build workflow from system .yaml description and set of templates.

    Args:
        system_infile (str): .yaml description of system. (default: system.yaml)
        build_dir (str): output directory for rendered templates. (default: build)
        template_dir (str): directory containing jinja2 template set. (default: templates)

    Returns:
        Nothing.
    """
    wfb = WorkflowBuilder(system_infile)
    wfb.template_dir = template_dir
    wfb.build_dir = build_dir
    wfb.initialize_template_engine()
    try:
        undefined_variables_output = wfb.show_undefined_variables()
    except Exception as e:
        print(e)
        error = e
        raise

    ### Conversion to tree with degenerate vertices
    wfb.descend()
    wfb.build_degenerate_graph()
    show_attributes_output = wfb.show_attributes()

    ## Build Workflow
    try:
        wfb.fill_templates()
    except Exception as e:
        print(e)
        error = e
        raise

    try:
        wf = wfb.compile_workflow()
    except Exception as e:
        print(e)
        error = e
        raise

    return

def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('system-infile',
        help='.yaml input file.', default='system.yaml')
    parser.add_argument('build-dir',
        help='output directory.', default='build')
    parser.add_argument('--template-dir',
        help="Directory containing templates.",
        default="templates")
    parser.add_argument('--verbose', '-v', action='store_true',
        help='Make this tool more verbose')
    parser.add_argument('--debug', action='store_true',
        help='Make this tool print debug info')
    args = parser.parse_args()

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING

    logger.setLevel(loglevel)

    logger.debug( args )

    logger.info("Build workflow from system desscription {} within output directory {} based on templates under {}".format(
        args.system_infile,
        args.build_dir,
        args.template_dir ) )
    build_wf(
        args.system,
        args.build_dir,
        args.template_dir )

if __name__ == '__main__':
    main()
