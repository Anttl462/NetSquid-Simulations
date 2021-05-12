import netsquid as ns
from netsquid.qubits.operators import create_rotation_op
import numpy as np
from scipy.stats import rv_discrete, poisson
import netsquid.components as models
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.instructions import INSTR_SWAP
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.qubits import ketstates as ks, QFormalism
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes.network import Network
from pydynaa import EventExpression
import netsquid.components.instructions as instr
from mip import Model, xsum, maximize, INTEGER
import os
import sys
import string


class EntangleNodes(NodeProtocol):

    def __init__(self, node, role, start_expression=None, input_mem_pos=0, num_pairs=1, name=None,
                 destination_probs=None):
        if role.lower() not in ['source', 'receiver']:
            raise ValueError

        self.tail = 0
        self.head = 0
        self.destination_dist = rv_discrete(
            values=(range(0, nlinks), list(destination_probs))) if destination_probs is not None else None

        self._is_source = role.lower() == 'source'
        self.name = name if name else 'EntangleNode({}, {})'.format(node.name, role)
        super().__init__(node=node, name=name)
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError('Start expression should be a {}, not a {}'.format(EventExpression, type(start_expression)))
        self.start_expression = start_expression
        self._num_pairs = num_pairs
        self._mem_positions = None

        # Claim input memory position:
        if self.node.qmemory is None:
            raise ValueError('Node {} does not have a quantum memory assigned.'.format(self.node))
        self._input_mem_pos = input_mem_pos
        self._qmem_input_port = self.node.qmemory.ports['qin{}'.format(self._input_mem_pos)]
        self.node.qmemory.mem_positions[self._input_mem_pos].in_use = True

    def start(self):
        self.entangled_pairs = 0  # counter
        self._mem_positions = [self._input_mem_pos]
        # Claim extra memory positions to use (if any):
        extra_memory = self._num_pairs - 1
        if extra_memory > 0:
            unused_positions = self.node.qmemory.unused_positions
            if extra_memory > len(unused_positions):
                raise RuntimeError('Not enough unused memory positions available: need {}, have {}'
                                   .format(self._num_pairs - 1, len(unused_positions)))
            for i in unused_positions[:extra_memory]:
                self._mem_positions.append(i)
                self.node.qmemory.mem_positions[i].in_use = True
        # Call parent start method
        return super().start()

    def stop(self):
        # Unclaim used memory positions:
        if self._mem_positions:
            for i in self._mem_positions[1:]:
                self.node.qmemory.mem_positions[i].in_use = False
            self._mem_positions = None
        # Call parent stop method
        super().stop()

    def run(self):
        # keep track of circular queue of qubits (this is subtracted from buffersize to get the actual index)

        while True:
            if self.start_expression is not None:
                yield self.start_expression
            elif self._is_source and self.entangled_pairs >= self._num_pairs:
                # If no start expression specified then limit generation to one round
                break
            # for mem_pos in self._mem_positions[::-1]:
            if True:
                # memory position to add new qubit into (buffersize - self.head) is head of queue
                mem_pos = buffersize - self.head - 1
                # Iterate in reverse so that input_mem_pos is handled last
                # if birthtime != -1
                if self._is_source:
                    self.node.subcomponents[self._qsource_name].trigger()
                yield self.await_port_input(self._qmem_input_port)

                if mem_pos != self._input_mem_pos:
                    self.node.qmemory.execute_instruction(
                        INSTR_SWAP, [self._input_mem_pos, mem_pos])
                    if self.node.qmemory.busy:
                        yield self.await_program(self.node.qmemory)
                # Update birthtime
                self.node.qmemory.qubit_birthtimes[mem_pos] = ns.sim_time()
                # (For stationary and MW) determine the destination of the qubit for e2ee:
                if self.destination_dist:  # (if using scheduler that pre-emptively matches)
                    self.node.qmemory.destinations[mem_pos] = self.destination_dist.rvs(size=1)
                # Move head to next position for next round
                self.head = (self.head + 1) % buffersize

                # For multiplexing
                self.entangled_pairs += 1

                # Signal to SwitchProtocol that LLE is done
                self.send_signal(Signals.SUCCESS, mem_pos)

    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        if self.node.qmemory is None:
            return False
        if self._mem_positions is None and len(self.node.qmemory.unused_positions) < self._num_pairs - 1:
            return False
        if self._mem_positions is not None and len(self._mem_positions) != self._num_pairs:
            return False
        if self._is_source:
            for name, subcomp in self.node.subcomponents.items():
                if isinstance(subcomp, QSource):
                    self._qsource_name = name
                    break
            else:
                return False
        return True


# The bread and butter of the simulation
class SwitchProtocol(LocalProtocol):
    # Protocol for a complete switch experiment

    def __init__(self, buffersize, nlinks, p, q, network, scheduler, workload, rounds, qubit_policy='oldest'):
        super().__init__(nodes=network.nodes,
                         name='Switch Experiment')
        if scheduler not in ['SJF', 'STCF', 'LJF', 'LTCF', 'FIFO', 'LIFO', 'OQF', 'YQF', 'Stationary', 'Max-Weight',
                             'Max-Weight-2']:
            raise ValueError

        if workload not in ['random', 'QKD', 'capacity', 'poisson', 'markov']:
            raise ValueError

        if rounds is None:
            raise ValueError

        # Run simulator for set number of rounds
        self.network = network
        self.nrounds = rounds
        self.nlinks = nlinks
        self.buffersize = buffersize
        self.workload = workload
        self.qubit_policy = qubit_policy
        self.scheduler = scheduler
        self.throughput = 0
        if T_0 != '':
            self.T = int(T_0)
        else:
            self.T = 1

        # Clock cycle length in nanoseconds
        self.cycle_length = 1000
        self.most_recent_cycle = 0

        # Default entanglement success probability is 1
        if p is None:
            p = np.full(nlinks, 1)
        self.p = p
        if q is None:
            q = 1
        self.q = q
        
        # For 'random' workload:
        # Heavy = 3
        # Light = 1
        self.max_num_jobs = 1
        self.max_job_size = 1

        # For 'Poisson': Heavy = 1.5, VeryHeavy = 1.6
        # if workload_size == 'Heavy':
        #     self.avg_num_jobs = 1.5
        # elif workload_size == 'VeryHeavy':
        #     self.avg_num_jobs = 1.6
        # else:
        #     self.avg_num_jobs = 1
        # self.avg_num_jobs = 3.726/2 

        if workload_size == 'Heavy':
            self.avg_num_jobs =  3.726/2
        elif workload_size == 'VeryHeavy':
            self.avg_num_jobs =  4.05/2
        else:
            self.avg_num_jobs = 2.43/2

        self.lambd = np.zeros([self.nlinks, self.nlinks])
        for x in range(0, nlinks):
            for y in range(0, nlinks):
                # Average rate
                if x != y:
                    self.lambd[x][y] = ((self.max_num_jobs / 2) * (np.average(range(1, self.max_job_size)))) / (nlinks-1)
                    self.lambd[x][y] = ((self.max_num_jobs / 2) * (                1                      )) / (nlinks-1)
        
        

        # arrival rate is uniform, pick one
        # Max job size is 1, just put 1 here
        # arrival_rate = np.average(range(1, self.max_job_size)) * (self.max_num_jobs / 2) * 2
        # arrival_rate =                      1                    * (self.max_num_jobs / 2) * 2
        # entanglement_rate = self.p[0]*q*self.nlinks
        # print('Arrival rate = '+str(arrival_rate))
        # print('Lambd rate = '+str(self.lambd[0][1]))
        # print('Entanglement rate = '+str(entanglement_rate))

        solution = np.full(self.nlinks, None)

        # Maximize F_ij (e2ee's) in eq \sum_{ij} U_ij * F_ij
        # Such that \sum_i F_ij \leq E_j for all j

        if self.scheduler == 'Stationary':
            # Normalization: For each link: p^0x_xy + p^0x_xz + ... <= 1
            # Constraints:  e2ee's need an LLE from both constituent links:
            #       p_x T p^0x_xy = p_y T p^0y_xy ---> p_x p^0x_xy - p_y p^0y_xy = 0
            # Stability: For each pair of links: p_x p^0x_xy q = \lambda_xy
            #           - (actually > \lambda_xy, but need to solve)
            #           - q = 1 for now... the BSM in the program always 'succeeds' but can give bad fidelity
            #            \sum_xy \lambda_xy < Total generated LLE's per unit time = \sum_x p_x q

            solution = np.zeros([self.nlinks, self.nlinks])
            for x in range(0, self.nlinks):
                for y in range(0, x):
                    solution[x][y] = self.lambd[x][y] / (self.p[x] * 1)  # 1 = q
                    solution[y][x] = solution[x][y] * (self.p[x] / self.p[y])
            # The total can be less than 1 (well, yes it can, but for simulation purposes it can't), so
            for x in range(0, self.nlinks):
                to_distribute = (1 - sum([solution[x][y] for y in range(0, self.nlinks)])) / (self.nlinks - 1)
                for y in range(0, self.nlinks):
                    if x != y:
                        solution[x][y] += to_distribute

        self.fidelities = np.full((nlinks, buffersize), 0.00000000000000000000000)

        self.job_queue = []
        self.req_matrix = np.zeros([self.nlinks, self.nlinks])

        

        self.data_manager_per_timestep = DataManager(id=path+'/'+'timestep_data'+str(sys.argv[1]), columns=['Buffer Occupancy', 'Throughput', 'Outstanding Requests'])
        self.data_manager_per_job = DataManager(id=path+'/'+'job_data'+str(sys.argv[1]), columns=['Response Time', 'Turnaround Time'])
        self.data_manager_per_e2ee = DataManager(id=path+'/'+'e2ee_data'+str(sys.argv[1]), columns=['Fidelity', 'User_1 Lifetime', 'User_2 Lifetime'])

        for i in range(0, nlinks):
            self.add_subprotocol(EntangleNodes(node=network.get_node('link_node_' + str(i)), role='source',
                                               name='entangle_link_' + str(i)))
            self.add_subprotocol(
                EntangleNodes(node=network.get_node('switch_node_' + str(i)), role='receiver',
                              name='entangle_switch_' + str(i), destination_probs=solution[i]))

        # Set entangle protocols start expressions
        for i in range(0, nlinks):
            start_expr_ent_A = (self.subprotocols['entangle_link_' + str(i)].await_signal(self, Signals.WAITING))
            self.subprotocols['entangle_link_' + str(i)].start_expression = start_expr_ent_A
            start_expr_ent_B = (self.subprotocols['entangle_switch_' + str(i)].await_signal(self, Signals.WAITING))
            self.subprotocols['entangle_switch_' + str(i)].start_expression = start_expr_ent_B

    def gen_kron(self, matrices):
        # Generalized Kronecker product for matrices = list of matrices.
        # e.g. if matrices = [X, Y, Z], then gen_kron(matrices) = XYZ (the Pauli operation)
        if len(matrices) == 1:
            return matrices[0]
        else:
            return self.gen_kron([np.kron(matrices[0], matrices[1]), ] + matrices[2:])

    def depolarization(self, num_qubits, fidelity):
        # num_qubits = number of qubits in the state
        # fidelity = fidelity of depolarized state
        # The depolarization operation is rho -> fidelity * rho + identity * (1-fidelity)/2**(num_qubits).
        # The Kraus operators for the completely depolarizing channel (fidelity = 0) are just a uniform
        # distribution of general Pauli operators.
        # Returns a depolarization function that you can apply to states.
        # The function takes a list of qubits.

        # Paulis.
        pauli_ns = [ns.I, ns.X, ns.Y, ns.Z]
        paulis = [np.array(p._matrix) for p in pauli_ns]

        # General Paulis.
        indices = lambda i, j: int(i / 4 ** j) % 4
        # indices(i, *) should give the i-th combination of Paulis in lexicographic order.
        # For instance, if num_qubits = 3, indices(4**3-1, *) = (3, 3, 3) --> (ZZZ)
        # Then indices(i, j) just gives the Pauli on the j-th qubit for the i-th general Pauli.
        gen_paulis = [self.gen_kron([paulis[indices(i, j)] for j in range(num_qubits)]) \
                      for i in range(4 ** num_qubits)]
        # Get operators.
        gen_pauli_ns = [ns.Operator('pauli' + str(i), gen_paulis[i]) for i in range(4 ** num_qubits)]
        # Kraus coefficients.
        coeffs = [(1. - fidelity) / 4 ** num_qubits] * (4 ** num_qubits)
        coeffs[0] += fidelity

        return lambda q: ns.qubits.qubitapi.multi_operate(q, gen_pauli_ns, coeffs)

    def BellStateMeasurement(self, q1, q2, total_gate_fidelity=1, meas_fidelity=1):
        # total_gate_fidelity = cumulative fidelity of two-qubit gates (CNOT, etc.).
        # meas_fidelity = fidelity of measuring the state of the electron spin.

        # Note that cnot_fidelity should include the effects of e.g. swapping.
        # Let F be the fidelity of a 2-qubit operation.
        # One of [atom1, atom2] is an electron spin; the other is a nuclear spin.
        # Then, we need two 2-qubit gates, giving a cumulative fidelity of F**2.
        # We also need to measure the electron spin twice.

        # Generate depolarization functions.
        gate_depol = self.depolarization(2, total_gate_fidelity)
        meas_depol = self.depolarization(1, meas_fidelity)

        # First do a CNOT on 1 and 2.
        ns.qubits.operate([q1, q2], ns.CNOT)

        # Apply the gate fidelity depolarization.
        gate_depol([q1, q2])

        # Rotate 1 by pi/2 about the y-axis.
        Y2 = ns.qubits.create_rotation_op(np.pi / 2, (0, 1, 0))
        ns.qubits.operate(q1, Y2)
        # Rotate 2 by pi about the y-axis.
        Y1 = ns.qubits.create_rotation_op(np.pi, (0, 1, 0))
        ns.qubits.operate(q2, Y1)

        # Apply depolarization noise to simulate measurement infidelities.
        meas_depol(q1)
        meas_depol(q2)

        result1 = ns.qubits.measure(q1, ns.Z)
        result2 = ns.qubits.measure(q2, ns.Z)

        # Remap results for later convenience.
        # newresult1 should tell us how many X gates to apply in order to successfully teleport
        # our qubit using BSM; newresult2 should tell us how many Z gates to apply.

        curr_result = (result1[0], result2[0])
        if curr_result == (0, 0):
            new_result = [(1, result1[1]), (1, result2[1])]
        elif curr_result == (1, 0):
            new_result = [(1, result1[1]), (0, result2[1])]
        elif curr_result == (0, 1):
            new_result = [(0, result1[1]), (1, result2[1])]
        elif curr_result == (1, 1):
            new_result = [(0, result1[1]), (0, result2[1])]

        return new_result, [q1, q2]

    # Update the Queue of Jobs according to the current workload configuration and scheduling policy
    # Job format:
    # [
    #   0: user 1,
    #   1: user 2,
    #   2: current size,
    #   3: initial size,
    #   4: boolean representing whether the job has been responded to,
    #   5: the round the job was birthed in
    #   6: the time the job was birthed in (ns)
    # ]
    def update_jobs(self, round_number):
        if self.workload == 'random':
            num_jobs = np.random.randint(self.max_num_jobs+1)
            while num_jobs > 0:
                # job_size = np.random.randint(self.max_job_size) + 1
                job_size = 1
                user_1 = user_2 = -1
                while user_1 == user_2:
                    user_1 = np.random.randint(self.nlinks)
                    user_2 = np.random.randint(self.nlinks)
                if self.req_matrix[min(user_1, user_2)][max(user_1, user_2)] >= 0:
                    job = [user_1, user_2, job_size, job_size, False, round_number, ns.sim_time()]
                    self.job_queue.append(job)
                else:
                    self.data_manager_per_job.record(np.array([0, 0]))
                    self.throughput += 1
                self.req_matrix[min(user_1, user_2)][max(user_1, user_2)] += job_size
                num_jobs -= 1

        elif self.workload == 'QKD':
            # Large sporadic bursts (effectively just random workload with at most 1 job per time slot) (NEEDS UPDATING)
            if np.random.randint(750) == 0:
                job_size = np.random.randint(500) + 500
                user_1 = user_2 = -1
                while user_1 == user_2:
                    user_1 = np.random.randint(self.nlinks)
                    user_2 = np.random.randint(self.nlinks)
                job = [user_1, user_2, job_size, job_size, False, round_number, ns.sim_time()]
                self.job_queue.append(job)
                self.req_matrix[min(user_1, user_2)][max(user_1, user_2)] += job_size
        
        elif self.workload == 'capacity':
            # Try to guarantee that there will be requests (keep the request queue at no more than 20) (NEEDS UPDATING)
            while sum([j[2] for j in self.job_queue]) < 20:
                job_size = np.random.randint(self.max_job_size) + 1
                user_1 = user_2 = -1
                while user_1 == user_2:
                    user_1 = np.random.randint(self.nlinks)
                    user_2 = np.random.randint(self.nlinks)
                job = [user_1, user_2, job_size, job_size, False, round_number, ns.sim_time()]
                if self.scheduler == 'LIFO':
                    self.job_queue.insert(0, job)
                else:
                    self.job_queue.append(job)
                self.req_matrix[min(job[0], job[1])][max(job[0], job[1])] += job[2]
        
        elif self.workload == 'poisson':
            num_jobs = poisson.rvs(self.avg_num_jobs)
            while num_jobs > 0:
                # job_size = np.random.randint(self.max_job_size) + 1
                job_size = 1
                user_1 = user_2 = -1
                while user_1 == user_2:
                    user_1 = np.random.randint(self.nlinks)
                    user_2 = np.random.randint(self.nlinks)
                if self.req_matrix[min(user_1, user_2)][max(user_1, user_2)] >= 0:
                    job = [user_1, user_2, job_size, job_size, False, round_number, ns.sim_time()]
                    self.job_queue.append(job)
                else:
                    self.data_manager_per_job.record(np.array([0, 0]))
                    self.throughput += 1
                self.req_matrix[min(user_1, user_2)][max(user_1, user_2)] += job_size
                num_jobs -= 1
        
        # Two-state markov workload (it's not iid but it's variance grows linearly)
        elif self.workload == 'markov':
            num_jobs = 0
            if np.random.rand() < 0.5:
                num_jobs = poisson.rvs(self.avg_num_jobs - self.avg_num_jobs/2)
            else:
                num_jobs = poisson.rvs(self.avg_num_jobs + self.avg_num_jobs/2)
            while num_jobs > 0:
                # job_size = np.random.randint(self.max_job_size) + 1
                job_size = 1
                user_1 = user_2 = -1
                while user_1 == user_2:
                    user_1 = np.random.randint(self.nlinks)
                    user_2 = np.random.randint(self.nlinks)
                if self.req_matrix[min(user_1, user_2)][max(user_1, user_2)] >= 0:
                    job = [user_1, user_2, job_size, job_size, False, round_number, ns.sim_time()]
                    self.job_queue.append(job)
                else:
                    self.data_manager_per_job.record(np.array([0, 0]))
                    self.throughput += 1
                self.req_matrix[min(user_1, user_2)][max(user_1, user_2)] += job_size
                num_jobs -= 1

        if self.scheduler == 'FIFO' or self.scheduler == 'LIFO':
            return
        if self.scheduler == 'SJF':
            # Sort by original size
            self.job_queue.sort(key=lambda x: x[3])
        if self.scheduler == 'STCF':
            # Sort by current size
            self.job_queue.sort(key=lambda x: x[2]) 
        if self.scheduler == 'LJF':
            # Sort by original size reversed
            self.job_queue.sort(key=lambda x: x[3])
            self.job_queue.reverse()
        if self.scheduler == 'LTCF':
            # Sort by current size reversed
            self.job_queue.sort(key=lambda x: x[2])
            self.job_queue.reverse()
        if self.scheduler == 'OQF':
            # Sort by average oldest qubit between the two parties in the job
            self.job_queue.sort(key=lambda x:
                (self.network.get_node('switch_node_' + str(x[0])).subcomponents[
                    'QuantumMemory' + str(x[0])].qubit_birthtimes[
                    self.network.get_node('switch_node_' + str(x[0])).subcomponents[
                        'QuantumMemory' + str(x[0])].get_oldest_qubit()] +
                self.network.get_node('switch_node_' + str(x[1])).subcomponents[
                    'QuantumMemory' + str(x[1])].qubit_birthtimes[
                    self.network.get_node('switch_node_' + str(x[1])).subcomponents[
                        'QuantumMemory' + str(x[1])].get_oldest_qubit()]) / 2)
            # Very Pythonic
        if self.scheduler == 'YQF':
            # Sort by average youngest qubit between the two parties in the job
            self.job_queue.sort(key=lambda x:
                (self.network.get_node('switch_node_' + str(x[0])).subcomponents[
                    'QuantumMemory' + str(x[0])].qubit_birthtimes[
                    self.network.get_node('switch_node_' + str(x[0])).subcomponents[
                        'QuantumMemory' + str(x[0])].get_youngest_qubit()] +
                self.network.get_node('switch_node_' + str(x[1])).subcomponents[
                    'QuantumMemory' + str(x[1])].qubit_birthtimes[
                    self.network.get_node('switch_node_' + str(x[1])).subcomponents[
                        'QuantumMemory' + str(x[1])].get_youngest_qubit()]) / 2)

        if self.scheduler == 'Stationary':
            # group the jobs up by which pair they're using (simulates separate queues per pair of links)
            self.job_queue.sort(key=lambda x: (x[0], x[1]))

        if self.scheduler == 'Max-Weight':
            if round_number % self.T == 0:
                # print('U: ' + str(self.req_matrix))

                m = Model('Max-Weight')
                x = [[m.add_var(var_type=INTEGER) for j in range(0, self.nlinks)] for i in range(0, self.nlinks)]
                m.objective = maximize(xsum(self.req_matrix[i][j] * x[i][j] for i in range(0, self.nlinks) for j in range(i+1, self.nlinks)))

                for i in range(0, self.nlinks):
                    I = list(range(0, self.nlinks))
                    I.remove(i)
                    # print("Link " + str(i) + " has " + str(network.get_node('switch_node_' + str(i)).qmemory.get_current_size()))
                    m += xsum(x[min(i, j)][max(i, j)] for j in I) <= \
                        network.get_node('switch_node_' + str(i)).qmemory.get_current_size()

                m.optimize()
                return [[x[i][j].x for j in range(0, self.nlinks)] for i in range(0, self.nlinks)]

        # alternative Max-Weight (needs updating)
        # if self.scheduler == 'Max-Weight 2':
        #     U = np.zeros([self.nlinks, self.nlinks])
        #     for job in self.job_queue:
        #         U[min(job[0], job[1])][max(job[0], job[1])] += job[2]
        #     m = Model('Max-Weight 2')
        #     x = [[m.add_var(var_type=INTEGER) for j in range(i, self.nlinks)] for i in range(0, self.nlinks)]
        #     m.objective = maximize(xsum(x[i][j] for i in range(0, self.nlinks) for j in range(i, self.nlinks)))
        #     for i in range(0, self.nlinks):
        #         m += xsum(x[i][j] for j in range(i, self.nlinks)) <= \
        #              network.get_node('switch_node_' + str(i)).qmemory.get_current_size()
        #         for j in range(i, self.nlinks):
        #             m += xsum(x[i][j] <= U[i][j])
        #     m.optimize()
        #     return [[x[i][j].x for j in range(i, self.nlinks)] for i in range(0, self.nlinks)]

    # Update the matrix of fidelities for monitoring
    def update_fidelities(self):
        for i in range(0, self.nlinks):
            for j in range(0, self.buffersize):
                q1, = network.get_node('link_node_' + str(i)).qmemory.peek(j)
                q2, = network.get_node('switch_node_' + str(i)).qmemory.peek(j)
                if q1 is not None and q2 is not None:
                    self.fidelities[i][j] = ns.qubits.fidelity([q1, q2], ks.b00)
                else:
                    self.fidelities[i][j] = 0

    def get_qubit_positions(self, user_1, user_2):
        # get initial value, then check if decohered
        pos_1 = pos_2 = -1
        if self.qubit_policy == 'oldest':
            pos_1 = network.get_node('switch_node_' + str(user_1)).subcomponents[
                'QuantumMemory' + str(user_1)].get_oldest_qubit()
            pos_2 = network.get_node('switch_node_' + str(user_2)).subcomponents[
                'QuantumMemory' + str(user_2)].get_oldest_qubit()
        elif self.qubit_policy == 'youngest':
            pos_1 = network.get_node('switch_node_' + str(user_1)).subcomponents[
                'QuantumMemory' + str(user_1)].get_youngest_qubit()
            pos_2 = network.get_node('switch_node_' + str(user_2)).subcomponents[
                'QuantumMemory' + str(user_2)].get_youngest_qubit()

        # If these qubits have decohered, update birthtimes and positions accordingly. Currently just
        # saying that if the fidelity of the link is < 0.6 it's decohered. I've been told this is a good metric.
        while self.fidelities[user_1][pos_1] < 0.6 and pos_1 != -1:
            network.get_node('switch_node_' + str(user_1)).subcomponents[
                'QuantumMemory' + str(user_1)].qubit_birthtimes[pos_1] = -1
            if self.qubit_policy == 'oldest':
                pos_1 = network.get_node('switch_node_' + str(user_1)).subcomponents[
                    'QuantumMemory' + str(user_1)].get_oldest_qubit()
            elif self.qubit_policy == 'youngest':
                pos_1 = network.get_node('switch_node_' + str(user_1)).subcomponents[
                    'QuantumMemory' + str(user_1)].get_youngest_qubit()
        while self.fidelities[user_2][pos_2] < 0.6 and pos_2 != -1:
            network.get_node('switch_node_' + str(user_2)).subcomponents[
                'QuantumMemory' + str(user_2)].qubit_birthtimes[pos_2] = -1
            if self.qubit_policy == 'oldest':
                pos_2 = network.get_node('switch_node_' + str(user_2)).subcomponents[
                    'QuantumMemory' + str(user_2)].get_oldest_qubit()
            elif self.qubit_policy == 'youngest':
                pos_2 = network.get_node('switch_node_' + str(user_2)).subcomponents[
                    'QuantumMemory' + str(user_2)].get_youngest_qubit()

        return [pos_1, pos_2]


    def run(self):
        self.start_subprotocols()
        rounds_left = self.nrounds
        network.get_node('test').subcomponents[
                                    'QuantumMemory'].execute_instruction(instr.INSTR_INIT, [0])
        while True:
            if ns.sim_time() - self.most_recent_cycle < self.cycle_length:
                print('bruh')
                network.get_node('test').subcomponents[
                                    'QuantumMemory'].execute_instruction(instr.IGate("Identity", create_rotation_op(0, (1, 0, 0))), [0])
                if network.get_node('test').qmemory.busy:
                    yield self.await_program(network.get_node('test').qmemory)
                continue
            self.most_recent_cycle = ns.sim_time()

            if rounds_left == 0:
                # Close the open files
                self.data_manager_per_timestep.close()
                self.data_manager_per_job.close()
                self.data_manager_per_e2ee.close()
                print('Worker '+str(sys.argv[1])+' Done!')
                break
            if rounds_left%20 == 0:
                # For debugging
                print('Worker '+str(sys.argv[1])+': '+str(rounds_left)+' rounds left, '+str(sum([j[2] for j in self.job_queue]))+' jobs in queue, '+str(sum([network.get_node('switch_node_' + str(i)).qmemory.get_current_size()
                                for i in range(0, self.nlinks)])) + ' qubits in memory')
            for i in range(0, self.nlinks):
                self.subprotocols['entangle_link_' + str(i)].entangled_pairs = 0
                self.subprotocols['entangle_switch_' + str(i)].entangled_pairs = 0

            # Wait for all links to generate their LLE's
            self.send_signal(Signals.WAITING)
            full_expr = self.await_signal(self.subprotocols['entangle_link_0'], Signals.SUCCESS)
            full_expr = full_expr & self.await_signal(self.subprotocols['entangle_switch_0'], Signals.SUCCESS)
            for i in range(1, self.nlinks):
                full_expr = full_expr & self.await_signal(self.subprotocols['entangle_link_' + str(i)], Signals.SUCCESS)
                full_expr = full_expr & self.await_signal(self.subprotocols['entangle_switch_' + str(i)],
                                                          Signals.SUCCESS)
            yield full_expr            

            # All links generated entanglements now
            # Apply probabilistic generation:
            for i in range(0, nlinks):
                mem_pos_1 = buffersize - self.subprotocols['entangle_switch_' + str(i)].head - 1
                mem_pos_2 = buffersize - self.subprotocols['entangle_link_' + str(i)].head - 1
                if np.random.rand() > self.p[i]:
                    # Throw out qubit (Do this, because if I limit the generation itself probabilistically (via the
                    # QSource), then this protocol won't know when to stop yielding, since the EntangleNodes protocol
                    # would never stop waiting for a qubit to arrive).

                    # Update birthtimes to indicate these are now gone
                    network.get_node('switch_node_' + str(i)).subcomponents[
                        'QuantumMemory' + str(i)].qubit_birthtimes[mem_pos_1] = -1
                    network.get_node('link_node_' + str(i)).subcomponents[
                        'QuantumMemory' + str(i)].qubit_birthtimes[mem_pos_2] = -1

                    # Pop the qubits from the switch memories:
                    network.get_node('switch_node_' + str(i)).qmemory.pop(mem_pos_1)
                    network.get_node('link_node_' + str(i)).qmemory.pop(mem_pos_2)

                    self.subprotocols['entangle_switch_' + str(i)].head = (self.subprotocols['entangle_switch_' + str(
                        i)].head - 1) % buffersize
                    self.subprotocols['entangle_link_' + str(i)].head = (self.subprotocols['entangle_link_' + str(
                        i)].head - 1) % buffersize
                else:
                    # Track the birthtimes now that these qubits have survived
                    network.get_node('switch_node_' + str(i)).subcomponents[
                        'QuantumMemory' + str(i)].qubit_birthtimes_cycles[mem_pos_1] = self.nrounds - rounds_left
                    network.get_node('link_node_' + str(i)).subcomponents[
                        'QuantumMemory' + str(i)].qubit_birthtimes_cycles[mem_pos_2] = self.nrounds - rounds_left

                    network.get_node('switch_node_' + str(i)).subcomponents[
                        'QuantumMemory' + str(i)].qubit_birthtimes[mem_pos_1] = ns.sim_time()
                    network.get_node('link_node_' + str(i)).subcomponents[
                        'QuantumMemory' + str(i)].qubit_birthtimes[mem_pos_2] = ns.sim_time()

            # Generate workload for this time step (use the 'solution' variable for the MW schedulers):
            solution = self.update_jobs(self.nrounds - rounds_left)
            # print(str(solution))
            # Job_queue is now updated and sorted according to scheduler

            if self.scheduler == 'Max-Weight' and (self.nrounds - rounds_left) % self.T != 0:
                self.data_manager_per_timestep.record(np.array([
                    # Occupancy:
                    sum([network.get_node('switch_node_' + str(i)).qmemory.get_current_size()
                                    for i in range(0, self.nlinks)]), 
                    # Throughput:
                    self.throughput, 
                    # Outstanding Requests:
                    sum([j[2] for j in self.job_queue])
                ]))
                self.throughput = 0
                rounds_left -= 1
                continue

            # Iterate through job queue
            job = 0
            while job < len(self.job_queue):
                # Get users from job queue
                user_1 = self.job_queue[job][0]
                user_2 = self.job_queue[job][1]
                if self.scheduler == 'Stationary':
                    # LLE's are matched pre-emptively, read the destinations from the switch buffer and check if user_1
                    # and user_2 are matched

                    # (if user_1 isn't destined in a user_2 LLE or user_2 isn't destined in a user_1 LLE)...
                    if (user_1 not in network.get_node('switch_node_' + str(user_2)).subcomponents[
                        'QuantumMemory' + str(user_2)].destinations) or (user_2 not in network.get_node(
                        'switch_node_' + str(user_1)).subcomponents['QuantumMemory' + str(user_1)].destinations):
                        # ... then, go on to the next job to try again
                        job += 1
                        continue

                if self.scheduler == 'Max-Weight' or self.scheduler == 'Max-Weight 2':
                    # Assign similarly to stationary, determined by the MW calculation from update_jobs
                    if solution[min(user_1, user_2)][max(user_1, user_2)] == 0:
                        job += 1
                        continue
                
                # Python doesn't have do-while loops, so I'll do this instead :/
                while True:
                    # Note that this will keep trying the same matching if there is a BSM failure
        
                    # Update the fidelities:
                    self.update_fidelities()

                    # Get positions of qubits according to qubit policy
                    [pos_1, pos_2] = self.get_qubit_positions(user_1, user_2)

                    if pos_1 == -1 or pos_2 == -1:
                        # No qubits available for these users. Move to next job
                        break

                    # Perform the entangling measurement
                    # Get switch qubits
                    q1 = network.get_node('switch_node_' + str(user_1)).qmemory.peek(pos_1)[0]
                    q2 = network.get_node('switch_node_' + str(user_2)).qmemory.peek(pos_2)[0]

                    # Run BSM
                    result, _ = self.BellStateMeasurement(q1, q2)

                    # Get link qubits
                    link_q1 = network.get_node('link_node_' + str(user_1)).qmemory.peek(pos_1)[0]
                    link_q2 = network.get_node('link_node_' + str(user_2)).qmemory.peek(pos_2)[0]

                    # Apply pauli operators
                    if result[0][0] == 1:
                        # Apply X to the qubit at the link node
                        network.get_node('link_node_' + str(user_2)).subcomponents[
                            'QuantumMemory' + str(user_2)].execute_instruction(instr.INSTR_X, [pos_2])
                        # Wait for processor if busy
                        if network.get_node('link_node_' + str(user_2)).qmemory.busy:
                            yield self.await_program(network.get_node('link_node_' + str(user_2)).qmemory)

                    if result[1][0] == 1:
                        # Apply Z to the qubit at the link node
                        network.get_node('link_node_' + str(user_2)).subcomponents[
                            'QuantumMemory' + str(user_2)].execute_instruction(instr.INSTR_Z, [pos_2])
                        # Wait for processor if busy
                        if network.get_node('link_node_' + str(user_2)).qmemory.busy:
                            yield self.await_program(network.get_node('link_node_' + str(user_2)).qmemory)

                    # Update list of fidelities, throughput, and lifetimes
                    fidelity = ns.qubits.fidelity([link_q1, link_q2], ks.b00)
                    
                    # Apply probabilistic bell state success:
                    if np.random.rand() < self.q:
                        self.throughput += 1

                        # Entanglement satisfied, update job queue and request matrix
                        self.job_queue[job][2] -= 1
                        self.req_matrix[min(user_1, user_2)][max(user_1, user_2)] -= 1

                        if self.scheduler == 'Max-Weight' or self.scheduler == 'Max-Weight 2':
                            solution[user_1][user_2] -= 1
                    
                        # print('e2e entanglement fidelity: '+str(fidelity))
                        self.data_manager_per_e2ee.record(np.array([
                            # Fidelity:
                            fidelity,
                            # # Lifetime 1:
                            # self.nrounds - rounds_left - network.get_node('switch_node_' + str(user_1)).subcomponents[
                            #     'QuantumMemory' + str(user_1)].qubit_birthtimes_cycles[pos_1], 
                            # # Lifetime 2:
                            # self.nrounds - rounds_left - network.get_node('switch_node_' + str(user_2)).subcomponents[
                            #     'QuantumMemory' + str(user_2)].qubit_birthtimes_cycles[pos_2]

                            
                            # Lifetime 1:
                            ns.sim_time() - network.get_node('switch_node_' + str(user_1)).subcomponents[
                                'QuantumMemory' + str(user_1)].qubit_birthtimes[pos_1], 
                            # Lifetime 2:
                            ns.sim_time() - network.get_node('switch_node_' + str(user_2)).subcomponents[
                                'QuantumMemory' + str(user_2)].qubit_birthtimes[pos_2]
                        ]))

                    # Update birthtimes to indicate these are now used
                    network.get_node('switch_node_' + str(user_1)).subcomponents[
                        'QuantumMemory' + str(user_1)].qubit_birthtimes[pos_1] = -1
                    network.get_node('switch_node_' + str(user_2)).subcomponents[
                        'QuantumMemory' + str(user_2)].qubit_birthtimes[pos_2] = -1

                    # Pop the qubits from the switch memories:
                    network.get_node('switch_node_' + str(user_1)).qmemory.pop(pos_1)
                    network.get_node('switch_node_' + str(user_2)).qmemory.pop(pos_2)
                    network.get_node('link_node_' + str(user_1)).qmemory.pop(pos_1)
                    network.get_node('link_node_' + str(user_2)).qmemory.pop(pos_2)

                    

                    # Check if this is the first response to this job:
                    if self.job_queue[job][4] == -1:
                        # Update response time in job data
                        self.job_queue[job][4] = self.nrounds - rounds_left - self.job_queue[job][5]

                    # Second condition
                    if self.job_queue[job][2] == 0:
                        # Job finished

                        # # Record reponse and turnaround times for this job (cycles)
                        # self.data_manager_per_job.record(np.array([
                        #     self.job_queue[job][4], 
                        #     self.nrounds - rounds_left - self.job_queue[job][5]
                        # ]))

                        # Record reponse and turnaround times for this job (ns)
                        self.data_manager_per_job.record(np.array([
                            self.job_queue[job][4], 
                            ns.sim_time() - self.job_queue[job][6]
                        ]))

                        # Pop job off queue and update the indices so things don't go bad
                        self.job_queue.pop(job)
                        job -= 1
                        break
                
                job += 1

            # SPECIAL LOOP JUST FOR MAX-WEIGHT:
            # Iterate through solution matrix and satisfy all extra e2ee's allotted by the scheduler
            if self.scheduler == "Max-Weight":
                for user_1 in range(0, self.nlinks):
                    for user_2 in range(user_1+1, self.nlinks):
                        # Assign similarly to stationary, determined by the MW calculation from update_jobs
                        if solution[min(user_1, user_2)][max(user_1, user_2)] == 0:
                            break
                        
                        while solution[user_1][user_2] > 0:
                            # Note that this will keep trying the same matching if there is a BSM failure, and will terminate once there are no qubits available
                
                            # Update the fidelities:
                            self.update_fidelities()

                            # Get positions of qubits according to qubit policy
                            [pos_1, pos_2] = self.get_qubit_positions(user_1, user_2)

                            if pos_1 == -1 or pos_2 == -1:
                                # No qubits available for these users. Move to next job
                                break

                            # Perform the entangling measurement
                            # Get switch qubits
                            q1 = network.get_node('switch_node_' + str(user_1)).qmemory.peek(pos_1)[0]
                            q2 = network.get_node('switch_node_' + str(user_2)).qmemory.peek(pos_2)[0]

                            # Run BSM
                            result, _ = self.BellStateMeasurement(q1, q2)

                            # Get link qubits
                            link_q1 = network.get_node('link_node_' + str(user_1)).qmemory.peek(pos_1)[0]
                            link_q2 = network.get_node('link_node_' + str(user_2)).qmemory.peek(pos_2)[0]

                            # Apply pauli operators
                            if result[0][0] == 1:
                                # Apply X to the qubit at the link node
                                network.get_node('link_node_' + str(user_2)).subcomponents[
                                    'QuantumMemory' + str(user_2)].execute_instruction(instr.INSTR_X, [pos_2])
                                # Wait for processor if busy
                                if network.get_node('link_node_' + str(user_2)).qmemory.busy:
                                    yield self.await_program(network.get_node('link_node_' + str(user_2)).qmemory)

                            if result[1][0] == 1:
                                # Apply Z to the qubit at the link node
                                network.get_node('link_node_' + str(user_2)).subcomponents[
                                    'QuantumMemory' + str(user_2)].execute_instruction(instr.INSTR_Z, [pos_2])
                                # Wait for processor if busy
                                if network.get_node('link_node_' + str(user_2)).qmemory.busy:
                                    yield self.await_program(network.get_node('link_node_' + str(user_2)).qmemory)

                            # Update list of fidelities, throughput, and lifetimes
                            fidelity = ns.qubits.fidelity([link_q1, link_q2], ks.b00)
                            
                            # find job this corresponds to:
                            job = 0
                            while job < len(self.job_queue) and self.job_queue[job][0] != user_1 and self.job_queue[job][1] != user_2:
                                job += 1
                                if job >= len(self.job_queue):
                                    job = -1
                                    break
                            
                            if len(self.job_queue) == 0:
                                job = -1
                            # Apply probabilistic bell state success:
                            if np.random.rand() < self.q:

                                # Entanglement satisfied, update job queue and request matrix
                                if job != -1:
                                    # print(self.job_queue)
                                    # print(job)
                                    self.job_queue[job][2] -= 1
                                self.req_matrix[min(user_1, user_2)][max(user_1, user_2)] -= 1

                                solution[user_1][user_2] -= 1
                            
                                # print('e2e entanglement fidelity: '+str(fidelity))
                                self.data_manager_per_e2ee.record(np.array([
                                    # Fidelity:
                                    fidelity,
                                    # # Lifetime 1:
                                    # self.nrounds - rounds_left - network.get_node('switch_node_' + str(user_1)).subcomponents[
                                    #     'QuantumMemory' + str(user_1)].qubit_birthtimes_cycles[pos_1], 
                                    # # Lifetime 2:
                                    # self.nrounds - rounds_left - network.get_node('switch_node_' + str(user_2)).subcomponents[
                                    #     'QuantumMemory' + str(user_2)].qubit_birthtimes_cycles[pos_2]

                                    # Lifetime 1:
                                    ns.sim_time() - network.get_node('switch_node_' + str(user_1)).subcomponents[
                                        'QuantumMemory' + str(user_1)].qubit_birthtimes[pos_1], 
                                    # Lifetime 2:
                                    ns.sim_time() - network.get_node('switch_node_' + str(user_2)).subcomponents[
                                        'QuantumMemory' + str(user_2)].qubit_birthtimes[pos_2]
                                        ]))

                            # Update birthtimes to indicate these are now used
                            network.get_node('switch_node_' + str(user_1)).subcomponents[
                                'QuantumMemory' + str(user_1)].qubit_birthtimes[pos_1] = -1
                            network.get_node('switch_node_' + str(user_2)).subcomponents[
                                'QuantumMemory' + str(user_2)].qubit_birthtimes[pos_2] = -1

                            # Pop the qubits from the switch memories:
                            network.get_node('switch_node_' + str(user_1)).qmemory.pop(pos_1)
                            network.get_node('switch_node_' + str(user_2)).qmemory.pop(pos_2)
                            network.get_node('link_node_' + str(user_1)).qmemory.pop(pos_1)
                            network.get_node('link_node_' + str(user_2)).qmemory.pop(pos_2)

                            # Check if this is the first response to this job:
                            if job != -1 and self.job_queue[job][4] == -1:
                                # # Update response time in job data (cycles)
                                # self.job_queue[job][4] = self.nrounds - rounds_left - self.job_queue[job][5]

                                # Update response time in job data (ns)
                                self.job_queue[job][4] = ns.sim_time() - self.job_queue[job][6]

                            # Second condition
                            if job != -1 and self.job_queue[job][2] == 0:
                                # Job finished

                                # # Record reponse and turnaround times for this job (cycles)
                                # self.data_manager_per_job.record(np.array([
                                #     self.job_queue[job][4], 
                                #     self.nrounds - rounds_left - self.job_queue[job][5]
                                # ]))

                                # Record reponse and turnaround times for this job (ns)
                                self.data_manager_per_job.record(np.array([
                                    self.job_queue[job][4], 
                                    ns.sim_time() - self.job_queue[job][6]
                                ]))
                                # Pop job off queue
                                self.job_queue.pop(job)
                        

            # After the 'while job < len(self.job_queue)' loop
            # Save Buffer Occupancy, Throughput, and Outstanding Requests data
            # print("Fidelities: "+str(self.fidelities))
            self.data_manager_per_timestep.record(np.array([
                # Occupancy:
                sum([network.get_node('switch_node_' + str(i)).qmemory.get_current_size()
                                for i in range(0, self.nlinks)]), 
                # Throughput:
                self.throughput, 
                # Outstanding Requests:
                sum([j[2] for j in self.job_queue])
            ]))
            self.throughput = 0
            rounds_left -= 1
    

# Helper class for recording data
class DataManager:

    def __init__(self, id, columns):
        self.file = open('NSSims/Sim_Data/'+str(id)+'.npy', 'wb')

    def record(self, data):
        np.save(self.file, data)
    
    def close(self):
        self.file.close()


class SwitchBuffer(QuantumProcessor):

    def __init__(self, name, num_positions, mem_noise_models, phys_instructions, fallback_to_nonphysical):
        # Just for consistency in variable names:
        self.buffersize = num_positions
        # Array to keep track of which qubits are the oldest/youngest
        self.qubit_birthtimes = np.full(num_positions, -1)
        # Keep track of the birthtime in cycles as well for data display
        self.qubit_birthtimes_cycles = np.full(num_positions, -1)
        # Keep track of the pre-emptive destination of each LLE (for use in Stationary and MW)
        self.destinations = np.full(num_positions, -1)
        super().__init__(name=name, num_positions=num_positions, mem_noise_models=mem_noise_models,
                         phys_instructions=phys_instructions, fallback_to_nonphysical=fallback_to_nonphysical)

    # Return index of oldest qubit in this buffer
    def get_oldest_qubit(self):
        index = -1
        # Looking for minimum birthtime
        curr_min = ns.sim_time()
        for i in range(0, self.buffersize):
            if curr_min >= self.qubit_birthtimes[i] >= 0:
                curr_min = self.qubit_birthtimes[i]
                index = i
        return index

    # Return index of youngest qubit in this buffer
    def get_youngest_qubit(self):
        index = -1
        # Looking for maximum birthtime
        curr_max = -1
        for i in range(0, self.buffersize):
            if curr_max <= self.qubit_birthtimes[i] >= 0:
                curr_max = self.qubit_birthtimes[i]
                index = i
        return index

    # Get the current size of the buffer
    def get_current_size(self):
        size = 0
        for time in self.qubit_birthtimes:
            if time != -1:
                size += 1
        return int(size)


def network_setup(nlinks, buffersize, prep_delay=0):
    network = Network('Entangle_nodes')

    link_nodes = network.add_nodes(['link_node_' + str(i) for i in range(0, nlinks)])
    switch_nodes = network.add_nodes(['switch_node_' + str(i) for i in range(0, nlinks)])
    

    # Page 11: https://arxiv.org/pdf/1809.00364.pdf (?)
    noise = models.T1T2NoiseModel(T1 = 6 * 10**(10), T2 = int(T2)) 
    # depolar = models.DepolarNoiseModel(time_independent=False, depolar_rate=1000000)
    # dephase = models.DephaseNoiseModel(time_independent=False, dephase_rate=1000000)

    phys_instructions = [PhysicalInstruction(instr.INSTR_SWAP, duration=1, quantum_noise_model=None),
                         PhysicalInstruction(instr.INSTR_X, duration=32, quantum_noise_model=None),
                         PhysicalInstruction(instr.INSTR_Z, duration=32, quantum_noise_model=None),
                         PhysicalInstruction(instr.IGate("Identity", create_rotation_op(0, (1, 0, 0))), duration=1000, quantum_noise_model=None)]

    test_node = network.add_node('test')
    test_node.add_subcomponent(SwitchBuffer(
            'QuantumMemory', 1, fallback_to_nonphysical=True,
            phys_instructions=phys_instructions,
            mem_noise_models=[noise] * 1))

    

    fibre_delay = models.FibreDelayModel()
    fixed_delay = models.FixedDelayModel(delay=prep_delay)


    for i in range(0, nlinks):
        print('Setting up link ' + str(i))
        link_nodes[i].add_subcomponent(SwitchBuffer(
            'QuantumMemory' + str(i), buffersize, fallback_to_nonphysical=True,
            phys_instructions=phys_instructions,
            mem_noise_models=[noise] * buffersize))
        switch_nodes[i].add_subcomponent(SwitchBuffer(
            'QuantumMemory' + str(i), buffersize, fallback_to_nonphysical=True,
            phys_instructions=phys_instructions,
            mem_noise_models=[noise] * buffersize))

        link_nodes[i].add_subcomponent(
            QSource('QSource' + str(i), state_sampler=StateSampler(qreprs=[ks.b00]),
                    num_ports=2, status=SourceStatus.EXTERNAL,
                    models={'emission_delay_model': fixed_delay}))
        # Create and connect quantum channel:
        qchannel = QuantumChannel('QuantumChannel' + str(i), length=1,
                                  models={'quantum_noise_model': noise,
                                          'delay_model': fibre_delay})
        port_name_a, port_name_b = network.add_connection(
            link_nodes[i], switch_nodes[i], channel_to=qchannel, label='connection' + str(i))
        # Setup Link ports:
        link_nodes[i].subcomponents['QSource' + str(i)].ports['qout0'].forward_output(
            link_nodes[i].ports[port_name_a])
        link_nodes[i].subcomponents['QSource' + str(i)].ports['qout1'].connect(
            link_nodes[i].qmemory.ports['qin0'])
        # Setup Switch ports:
        switch_nodes[i].ports[port_name_b].forward_input(switch_nodes[i].qmemory.ports['qin0'])
    return network


if __name__ == '__main__':
    ns.set_qstate_formalism(QFormalism.DM)
    ns.sim_reset()
    
    nlinks = int(sys.argv[2])
    buffersize = int(sys.argv[3])
    scheduler = sys.argv[4]
    p = np.full(nlinks, float(sys.argv[5]))
    q = float(sys.argv[6])
    nrounds = int(sys.argv[7])
    qubit_policy = sys.argv[8]
    workload_size = sys.argv[9]
    T_0 = sys.argv[10] if scheduler == "Max-Weight" else ''
    run_over = sys.argv[11] if scheduler == "Max-Weight" else sys.argv[10]
    T2 = sys.argv[12] if scheduler == "Max-Weight" else sys.argv[11]

    path = workload_size + '_' + scheduler + '_' + qubit_policy[0].upper()
    if scheduler == "Max-Weight":
        path += '_' + str(T_0)
    path += '_' + str(buffersize)
    path += '_' + str(p[0])
    path += '_' + str(q)
    path += '_' + str(T2)
    path += '_' + str(run_over)
    path += '_' + str(nlinks)
    


    print("nlinks = "+str(nlinks))
    print(path)
    print(p)

    network = network_setup(nlinks, buffersize)
    switch = SwitchProtocol(buffersize=buffersize, nlinks=nlinks, p=p, q=q, network=network, scheduler=scheduler,
                            workload='markov',
                            rounds=nrounds, qubit_policy=qubit_policy)

    switch.start()
    ns.sim_run()
