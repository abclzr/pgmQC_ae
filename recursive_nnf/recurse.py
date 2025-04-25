import networkx as nx

from qiskit.dagcircuit.dagnode import DAGOpNode, DAGOutNode
from qiskit.converters import dag_to_circuit

from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination

import nnf
import pydot

tensor_mode = False

def recurse (dag, bn, query_vars, depth):

    # print(bn)
    # ve = VariableElimination(bn)

    # from pgmpy.inference.EliminationOrder import MinFill
    # mfeo = MinFill(bn).get_elimination_order()
    # for thing in mfeo:
    #     print("mfeo")
    #     print(thing)
    #     print(MinFill(bn).cost(thing))
    # mfig = ve.induced_graph(mfeo)
    # for node in mfig.nodes():
    #     print("node in mfig.nodes")
    #     print(node)
    # for edge in mfig.edges():
    #     print("edge in mfig.edges()")
    #     print(edge)
    # nx.draw(mfig)
    # import matplotlib.pyplot as plt
    # plt.show()
    # print(ve.induced_width(mfeo))

    print("depth")
    print(depth)

    dag.draw(filename="dag.png")
    circuit = dag_to_circuit(dag)
    print(circuit)

    # RECURSIVE CASE: FIND CUTS
    mg = nx.MultiGraph(dag.edges())
    rem_nodes = []
    for node in mg.nodes():
        if not isinstance(node,DAGOpNode):
            rem_nodes.append(node)
    mg.remove_nodes_from(rem_nodes)

    recurse_cut_sizes = []
    if depth<4 and 1<len(mg):

        partitions = nx.community.edge_current_flow_betweenness_partition(G=mg,number_of_sets=2)
        assert len(partitions)>1

        query_nodes = []
        for node in bn.nodes():
            src_part = None
            dst_part = None
            for num, partition in enumerate(partitions):
                if node[0] in partition:
                    src_part = num
                if node[1] in partition:
                    dst_part = num
            assert src_part!=None or dst_part!=None
            if src_part!=None and dst_part!=None and src_part!=dst_part:
                query_nodes.append(node)

        recurse_cut_sizes = [len(query_nodes)]

        if len(query_nodes)>0:

            # RECURSIVE CASE: RECURSIVE CALL
            query_vars_recurse = query_vars.copy()
            query_vars_recurse.extend(query_nodes)

            factors = []
            clauses = []
            weights = {}

            for partition in partitions:

                # Manage the Qiskit QuantumCircuit
                dag_recurse = dag.copy_empty_like() # get the metadata
                dag_recurse._multi_graph = dag._multi_graph.copy()
                dag_recurse._op_names = dag._op_names.copy()

                for op_node in dag_recurse.op_nodes():
                    if op_node not in partition:
                        dag_recurse.remove_op_node(op_node)

                # Manage the pgmpy BayesianNetwork
                bn_cut = bn.copy()
                rem_nodes = []
                for node in bn_cut:
                    if node in query_nodes:
                        continue
                    elif node[0] in partition:
                        continue
                    elif node[1] in partition:
                        continue
                    else:
                        rem_nodes.append(node)
                bn_cut.remove_nodes_from(rem_nodes)

                recurse_cut_size, factor, ac_tuples = recurse (dag_recurse, bn_cut, [var for var in query_vars_recurse if var not in rem_nodes], depth+1)
                recurse_cut_sizes.append(recurse_cut_size)
                factors.append(factor)

                # Manage the nnf arithmetic circuit
                uids = []
                for fSet in ac_tuples:

                    qvp = {}
                    for qv in [var for var in query_vars_recurse if var not in rem_nodes]:
                        qvp[qv] = "I"

                    for tup in fSet:
                        qubit_state = tup[0]
                        qubit_pauli = tup[1]
                        if   qvp[qubit_state]=="X" and qubit_pauli=="Z":
                            qvp[qubit_state] = "Y"
                        elif qvp[qubit_state]=="Z" and qubit_pauli=="X":
                            qvp[qubit_state] = "Y"
                        else:
                            qvp[qubit_state] = qubit_pauli

                    # implies = []
                    implied_by = []
                    for key in qvp:
                        qubit_state = key
                        qubit_pauli = qvp[key]

                        # inclusion = []
                        # for pauli1 in ["I","X","Y","Z"]:
                        #     inclusion.append(nnf.Var((qubit_state,pauli1),true=True))
                        #     for pauli2 in ["I","X","Y","Z"]:
                        #         if pauli1 != pauli2:
                        #             clauses.append(nnf.Or([ nnf.Var((qubit_state,pauli1),true=False), nnf.Var((qubit_state,pauli2),true=False) ]))
                        # clauses.append(nnf.Or(inclusion))

                        # for pauli in ["I","X","Y","Z"]:
                        #     # implies.append(nnf.Var((qubit_state,pauli),true=not(qubit_pauli==pauli)))
                        #     implied_by.append(nnf.Var((qubit_state,pauli),true=qubit_pauli==pauli))

                        implied_by.append(nnf.Var((qubit_state,"X"),true=qubit_pauli=="X" or qubit_pauli=="Y"))
                        implied_by.append(nnf.Var((qubit_state,"Z"),true=qubit_pauli=="Z" or qubit_pauli=="Y"))

                    import uuid
                    uid = uuid.uuid1()
                    uids.append(uid)
                    weights[uid] = ac_tuples[fSet]

                    # implies.append(nnf.Var(uid,true=True))
                    # clauses.append(nnf.Or(implies))

                    for node in implied_by:
                        clauses.append(nnf.Or([ nnf.Var(uid,true=False), node ]))

                inclusion = []
                for uid1 in uids:
                    inclusion.append(nnf.Var(uid1,true=True))
                    # for uid2 in uids:
                    #     if uid1 != uid2:
                    #         clauses.append(nnf.Or([ nnf.Var(uid1,true=False), nnf.Var(uid2,true=False) ]))
                clauses.append(nnf.Or(inclusion))

                # cnf = nnf.And(clauses)
                # for num, model in enumerate(cnf.models()):
                #     print (cnf.model_count())
                #     print (num)
                #     for key in model:
                #         if model[key]:
                #             print (key)

            assert(len(factors)==2)

            # RECURSIVE CASE: COMBINE RESULTS

            if (tensor_mode):
                from pgmpy.models import FactorGraph
                fg = FactorGraph()
                fg.add_nodes_from(query_vars_recurse)
                for factor in factors:
                    fg.add_factors(factor)
                    fg.add_edges_from([(var,factor) for var in factor.variables])
                print("to_markov_model")
                mm = fg.to_markov_model()
                print("VariableElimination")
                ve = VariableElimination(mm)

            cnf = nnf.And(clauses)

            print("compiling")
            print("depth")
            print(depth)
            print(circuit)
            print(recurse_cut_sizes)
            print("query_vars")
            print(len(query_vars))
            print("query_nodes")
            print(len(query_nodes))
            ac = nnf.dsharp.compile(cnf, executable="/common/home/yh804/research/dsharp/dsharp")
            # ac = cnf
            print("model_count")
            print(ac.model_count())
            print("assert(ac.decomposable())")
            # assert(ac.decomposable())
            # assert(ac.deterministic())
            # ac.mark_deterministic()
            print("Forgetting")
            import sys
            print("sys.getrecursionlimit()")
            print(sys.getrecursionlimit())
            print(sys.setrecursionlimit(4096))
            print(sys.getrecursionlimit())
            for node in query_nodes:
                for pauli in ['I','X','Y','Z']:
                    print(node)
                    print(pauli)
                    ac = ac.forget([(node,pauli)])
            # print("to_DOT")
            # pydot.graph_from_dot_data(ac.to_DOT(color=True))[0].write_png("contraction_ac.png")

            print("QWMC_merge")
            ac_tuples = nnf.amc.QWMC_merge( node=ac, vars={}, probs=weights )
            print(len(ac_tuples))

            run_base_case = False

        else:
            print("Cut size zero")
            run_base_case = True

    else:
        print("Graph size one")
        run_base_case = True

    if run_base_case:

        # BASE CASE: EVALUATE VIA ARITMETIC CIRCUIT
        from pgmpy.readwrite import NETWriter
        import os

        # NET
        print("writing bayesian network to file")
        NETWriter(bn).write_net(filename='circuit.net')

        garbled_node_to_node = {}
        for node in bn:
            garbled_node=str(node).replace("-","_").replace("=","_").replace(",","_").replace(" ","_").replace("(","_").replace(")","_").replace("'","_").replace("[","_").replace("]","_")+str(id(node))
            garbled_node_to_node[garbled_node] = node

        # path_to_bayes_to_cnf = "/common/home/yh804/research/bayes-to-cnf"
        # stdout = os.system(path_to_bayes_to_cnf + '/bin/bn-to-cnf -e -d -a -i circuit.net -w -s')
        path_to_qACE = "/common/home/yh804/research/ace_v3.0_linux86/"
        stdout = os.system(path_to_qACE + 'compile -cd06 -forceC2d -encodeOnly circuit.net')

        var_num_to_node_pauli = {}
        var_num_to_weight = {}
        forget_var_num = []
        with open('circuit.net.lmap', 'r') as lmap_file:
            for line in lmap_file:
                if line.startswith('cc'):
                    words = line.split('$')
                    if words[1] == 'K' or words[1] == 'S' or words[1] == 'N' or words[1] == 'v' or words[1] == 't':
                        continue
                    elif words[1] == 'V' or words[1] == 'T':
                        continue
                    elif words[1] == 'I':
                        var_num = int(words[2])
                        garbled_node = words[5]
                        node = garbled_node_to_node[garbled_node]
                        pauli = words[6].replace('0\n','I').replace('1\n','X').replace('2\n','Y').replace('3\n','Z')
                        if node in query_vars:
                            var_num_to_node_pauli[var_num] = (node,pauli)
                        else:
                            forget_var_num.append(var_num)
                        continue
                    elif words[1] == 'C':
                        var_num = int(words[2])
                        var_num_to_weight[var_num] = float(words[3])
                        continue

        with open('forget.txt', 'w') as forget_file:
            for var_num in forget_var_num:
                forget_file.write(str(var_num))
                forget_file.write(" ")

        # Compile with "/common/home/yh804/research/ace_v3.0_linux86/c2d_linux -reduce -in circuit.net.cnf -dt_in circuit.net.dtree -minimize -suppress_ane -determined circuit.net.pmap"
        stdout = os.system(path_to_qACE + 'c2d_linux -reduce -in circuit.net.cnf -dt_in circuit.net.dtree -minimize -determined circuit.net.pmap') # -visualize

        # with open('circuit.net.cnf','r') as cnf_file:
        #     ac = nnf.dsharp.compile(nnf.dimacs.load(cnf_file), executable="/common/home/yh804/research/dsharp/dsharp")
        #     assert(ac.marked_deterministic())
        #     # ac = ac.make_smooth()

        with open('circuit.net.cnf.nnf','r') as nnf_file:
            ac = nnf.dsharp.load(nnf_file)
            ac.mark_deterministic()
            assert(ac.decomposable())
            # assert(ac.deterministic())
            assert(ac.marked_deterministic())
            # ac = ac.make_smooth()

        # ac = ac.forget(forget_var_num)
        # pydot.graph_from_dot_data(ac.to_DOT(color=True))[0].write_png("ac.png")
        ac_tuples = nnf.amc.QWMC( node=ac, vars=var_num_to_node_pauli, probs=var_num_to_weight )

        ve = VariableElimination(bn)

    factor = None
    if (tensor_mode):
        bn_tuples = {}
        def string_value ( factor: TabularCPD, query_vars: [], pauli_string: {} ):

            if len(query_vars)==0:

                if factor.get_value():
                    bn_tuples[frozenset(pauli_string.items())] = factor.get_value()
                return

            for pauli in ['I','X','Y','Z']:
                pauli_string[query_vars[0]] = pauli
                string_value (
                    factor.reduce([(query_vars[0],pauli)],inplace=False),
                    query_vars[1:],
                    pauli_string.copy()
                )

        factor = ve.query(query_vars)
        string_value ( factor=factor, query_vars=query_vars, pauli_string={} )

        for key in ac_tuples:
            if 0.00000005<abs(ac_tuples[key]):
                assert( abs(bn_tuples[key]-ac_tuples[key])<0.00000005 )

        for key in bn_tuples:
            if 0.00000005<abs(bn_tuples[key]):
                assert( abs(bn_tuples[key]-ac_tuples[key])<0.00000005 )

    return recurse_cut_sizes, factor, ac_tuples