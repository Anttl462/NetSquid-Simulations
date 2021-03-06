import netsquid as ns
import numpy as np
import scipy
import scipy.stats
from netsquid.components import DepolarNoiseModel, FibreLossModel, T1T2NoiseModel, DephaseNoiseModel
from netsquid.components.component import Message, Port
from netsquid.components.qmemory import QuantumMemory
from netsquid.nodes import Node
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.instructions import INSTR_SWAP
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction, QuantumProgram
from netsquid.qubits import ketstates as ks, QFormalism
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes.network import Network
from pydynaa import EventExpression
import netsquid.components.instructions as instr
import matplotlib.pyplot as plt
import copy

from scipy.stats import rv_discrete


class EntangleNodes(NodeProtocol):

    def __init__(self, node, role, start_expression=None, input_mem_pos=0, num_pairs=1, name=None,
                 destination_probs=None):
        if role.lower() not in ["source", "receiver"]:
            raise ValueError

        # print(list(range(0, nlinks)))
        # print(destination_probs)
        self.destination_dist = rv_discrete(
            values=(range(0, nlinks), list(destination_probs))) if destination_probs is not None else None

        self._is_source = role.lower() == "source"
        self.name = name if name else "EntangleNode({}, {})".format(node.name, role)
        super().__init__(node=node, name=name)
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self.start_expression = start_expression
        self._num_pairs = num_pairs
        self._mem_positions = None
        # Claim input memory position:
        if self.node.qmemory is None:
            raise ValueError("Node {} does not have a quantum memory assigned.".format(self.node))
        self._input_mem_pos = input_mem_pos
        self._qmem_input_port = self.node.qmemory.ports["qin{}".format(self._input_mem_pos)]
        self.node.qmemory.mem_positions[self._input_mem_pos].in_use = True

    def start(self):
        self.entangled_pairs = 0  # counter
        self._mem_positions = [self._input_mem_pos]
        # Claim extra memory positions to use (if any):
        extra_memory = self._num_pairs - 1
        if extra_memory > 0:
            unused_positions = self.node.qmemory.unused_positions
            if extra_memory > len(unused_positions):
                raise RuntimeError("Not enough unused memory positions available: need {}, have {}"
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
        self.head = 0
        self.tail = 0
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

    def __init__(self, buffersize, nlinks, p, network, scheduler, workload, rounds, qubit_policy="oldest"):
        super().__init__(nodes=network.nodes,
                         name="Switch Experiment")
        if scheduler not in ["SJF", "STCF", "LJF", "LTCF", "FIFO", "OQF", "YQF", "Stationary"]:
            raise ValueError

        if workload not in ["random", "QKD"]:
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

        # Default entanglement success probability is 1
        if p is None:
            p = np.full(nlinks, 1)
        self.p = p

        # For "random" workload:
        self.max_num_jobs = 5
        self.max_job_size = 1
        self.lambd = np.zeros([self.nlinks, self.nlinks])
        for x in range(0, nlinks):
            for y in range(0, nlinks):
                # Average rate
                if x != y:
                    self.lambd[x][y] = ((self.max_num_jobs / 2) * (self.max_job_size + 1) / 2) / ((nlinks-1) ** 2)

        print(self.lambd)

        if self.scheduler == "Stationary":
            # Normalization: For each link: p^0x_xy + p^0x_xz + ... <= 1
            # Constraints:  e2ee's need an LLE from both constituent links:
            #       p_x T p^0x_xy = p_y T p^0y_xy ---> p_x p^0x_xy - p_y p^0y_xy = 0
            # Stability: For each pair of links: p_x p^0x_xy q = \lambda_xy
            #           - (actually > \lambda_xy, but need to solve)
            #           - q = 1 for now... the BSM in the program always "succeeds" but can give bad fidelity
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

        print(solution)
        # For debugging:
        self.fidelities = np.full((nlinks, buffersize), 0.00000000000000000000000)

        # Queue of jobs Each job is (user_1, user_2, current job size, original job size, time when first responded,
        # birth time)

        # Queue will be ordered based on scheduler (Shortest Job First (SJF), Shortest Time to Completion First (
        # STCF), Longest Job First (LJF), Longest Time to Completion First (LTCF), First in First Out (FIFO),
        # Oldest Qubit First (OQF), Youngest Qubit First (YCF))
        self.job_queue = []

        # Lists to use later in matplotlib
        self.e2e_fidelities = []
        self.eq = np.zeros(self.nrounds)
        self.throughput = np.zeros(self.nrounds)
        self.lifetimes = []
        self.turnaround_times = []
        self.response_times = []
        self.outstanding_requests = []

        for i in range(0, nlinks):
            self.add_subprotocol(EntangleNodes(node=network.get_node("link_node_" + str(i)), role="source",
                                               name="entangle_link_" + str(i)))
            self.add_subprotocol(
                EntangleNodes(node=network.get_node("switch_node_" + str(i)), role="receiver",
                              name="entangle_switch_" + str(i), destination_probs=solution[i]))

        # Set entangle protocols start expressions
        for i in range(0, nlinks):
            start_expr_ent_A = (self.subprotocols["entangle_link_" + str(i)].await_signal(self, Signals.WAITING))
            self.subprotocols["entangle_link_" + str(i)].start_expression = start_expr_ent_A
            start_expr_ent_B = (self.subprotocols["entangle_switch_" + str(i)].await_signal(self, Signals.WAITING))
            self.subprotocols["entangle_switch_" + str(i)].start_expression = start_expr_ent_B

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

    def BellStateMeasurement(self, q1, q2, total_gate_fidelity=0.9, meas_fidelity=0.9):
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
    # ]
    def update_jobs(self, round_number):
        if self.workload == "random":
            num_jobs = np.random.randint(self.max_num_jobs)
            while num_jobs > 0:
                # job_size = np.random.randint(max_job_size) + 1
                job_size = 1
                user_1 = user_2 = -1
                while user_1 == user_2:
                    user_1 = np.random.randint(self.nlinks)
                    user_2 = np.random.randint(self.nlinks)
                self.job_queue.append([user_1, user_2, job_size, job_size, False, round_number])
                num_jobs -= 1

        elif self.workload == "QKD":
            # Large sporadic bursts (effectively just random workload with at most 1 job per time slot)
            if np.random.randint(750) == 0:
                job_size = np.random.randint(500) + 500
                user_1 = user_2 = -1
                while user_1 == user_2:
                    user_1 = np.random.randint(self.nlinks)
                    user_2 = np.random.randint(self.nlinks)
                self.job_queue.append([user_1, user_2, job_size, job_size, False, round_number])

        if self.scheduler == "FIFO":
            return
        if self.scheduler == "SJF":
            # Sort by original size
            self.job_queue.sort(key=lambda x: x[3])
        if self.scheduler == "STCF":
            # Sort by current size
            self.job_queue.sort(key=lambda x: x[2])
        if self.scheduler == "LJF":
            # Sort by original size reversed
            self.job_queue.sort(key=lambda x: x[3])
            self.job_queue.reverse()
        if self.scheduler == "LTCF":
            # Sort by current size reversed
            self.job_queue.sort(key=lambda x: x[2])
            self.job_queue.reverse()
        if self.scheduler == "OQF":
            # Sort by average oldest qubit between the two parties in the job
            self.job_queue.sort(key=lambda x:
                (self.network.get_node("switch_node_" + str(x[0])).subcomponents[
                    "QuantumMemory" + str(x[0])].qubit_birthtimes[
                    self.network.get_node("switch_node_" + str(x[0])).subcomponents[
                        "QuantumMemory" + str(x[0])].get_oldest_qubit()] +
                    self.network.get_node("switch_node_" + str(x[1])).subcomponents[
                        "QuantumMemory" + str(x[1])].qubit_birthtimes[
                        self.network.get_node("switch_node_" + str(x[1])).subcomponents[
                            "QuantumMemory" + str(x[1])].get_oldest_qubit()])/2)
            # Very Pythonic
        if self.scheduler == "YQF":
            # Sort by average youngest qubit between the two parties in the job
            self.job_queue.sort(key=lambda x:
            (self.network.get_node("switch_node_" + str(x[0])).subcomponents[
                 "QuantumMemory" + str(x[0])].qubit_birthtimes[
                 self.network.get_node("switch_node_" + str(x[0])).subcomponents[
                     "QuantumMemory" + str(x[0])].get_youngest_qubit()] +
             self.network.get_node("switch_node_" + str(x[1])).subcomponents[
                 "QuantumMemory" + str(x[1])].qubit_birthtimes[
                 self.network.get_node("switch_node_" + str(x[1])).subcomponents[
                     "QuantumMemory" + str(x[1])].get_youngest_qubit()]) / 2)

        if self.scheduler == "Stationary":
            # group the jobs up by which pair they're using (simulates separate queues per pair of links)
            self.job_queue.sort(key=lambda x: (x[0], x[1]))

    # Update the matrix of fidelities for monitoring
    def update_fidelities(self):
        for i in range(0, self.nlinks):
            for j in range(0, self.buffersize):
                q1, = network.get_node("link_node_" + str(i)).qmemory.peek(j)
                q2, = network.get_node("switch_node_" + str(i)).qmemory.peek(j)
                if q1 is not None and q2 is not None:
                    self.fidelities[i][j] = ns.qubits.fidelity([q1, q2], ks.b00)
                else:
                    self.fidelities[i][j] = 0

    def get_qubit_positions(self, user_1, user_2):
        # get initial value, then check if decohered
        if self.qubit_policy == "oldest":
            pos_1 = network.get_node("switch_node_" + str(user_1)).subcomponents[
                "QuantumMemory" + str(user_1)].get_oldest_qubit()
            pos_2 = network.get_node("switch_node_" + str(user_2)).subcomponents[
                "QuantumMemory" + str(user_2)].get_oldest_qubit()
        elif self.qubit_policy == "youngest":
            pos_1 = network.get_node("switch_node_" + str(user_1)).subcomponents[
                "QuantumMemory" + str(user_1)].get_youngest_qubit()
            pos_2 = network.get_node("switch_node_" + str(user_2)).subcomponents[
                "QuantumMemory" + str(user_2)].get_youngest_qubit()

        # If these qubits have decohered, update birthtimes and positions accordingly. Currently just
        # saying that if the fidelity of the link is < 0.6 it's decohered. I've been told this is a good metric.
        while self.fidelities[user_1][pos_1] < 0.6 and pos_1 != -1:
            network.get_node("switch_node_" + str(user_1)).subcomponents[
                "QuantumMemory" + str(user_1)].qubit_birthtimes[pos_1] = -1
            if self.qubit_policy == "oldest":
                pos_1 = network.get_node("switch_node_" + str(user_1)).subcomponents[
                    "QuantumMemory" + str(user_1)].get_oldest_qubit()
            elif self.qubit_policy == "youngest":
                pos_1 = network.get_node("switch_node_" + str(user_1)).subcomponents[
                    "QuantumMemory" + str(user_1)].get_youngest_qubit()
        while self.fidelities[user_2][pos_2] < 0.6 and pos_2 != -1:
            network.get_node("switch_node_" + str(user_2)).subcomponents[
                "QuantumMemory" + str(user_2)].qubit_birthtimes[pos_2] = -1
            if self.qubit_policy == "oldest":
                pos_2 = network.get_node("switch_node_" + str(user_2)).subcomponents[
                    "QuantumMemory" + str(user_2)].get_oldest_qubit()
            elif self.qubit_policy == "youngest":
                pos_2 = network.get_node("switch_node_" + str(user_2)).subcomponents[
                    "QuantumMemory" + str(user_2)].get_youngest_qubit()

        return [pos_1, pos_2]

    def run(self):
        self.start_subprotocols()
        rounds_left = self.nrounds
        while True:
            if rounds_left == 0:
                # pass finished data to visualizer
                # (self, fidelities, throughput, eq, lifetimes, turnarounds, responses):
                plotter = Plotter(scheduler=self.scheduler, qubit_scheduler=self.qubit_policy,
                                  fidelities=self.e2e_fidelities, throughput=self.throughput, eq=self.eq,
                                  lifetimes=self.lifetimes, turnarounds=self.turnaround_times,
                                  responses=self.response_times, outstanding_requests=self.outstanding_requests)
                break
            print("-------------------------------------------------------------------")
            print("Round = ", rounds_left, ", Time =", ns.sim_time())
            for i in range(0, self.nlinks):
                self.subprotocols["entangle_link_" + str(i)].entangled_pairs = 0
                self.subprotocols["entangle_switch_" + str(i)].entangled_pairs = 0

            # Wait for all links to generate their LLE's
            self.send_signal(Signals.WAITING)
            full_expr = self.await_signal(self.subprotocols["entangle_link_0"], Signals.SUCCESS)
            full_expr = full_expr & self.await_signal(self.subprotocols["entangle_switch_0"], Signals.SUCCESS)
            for i in range(1, self.nlinks):
                full_expr = full_expr & self.await_signal(self.subprotocols["entangle_link_" + str(i)], Signals.SUCCESS)
                full_expr = full_expr & self.await_signal(self.subprotocols["entangle_switch_" + str(i)],
                                                          Signals.SUCCESS)
            yield full_expr

            # All links generated entanglements now
            # Apply probabilistic generation:
            for i in range(0, nlinks):
                mem_pos_1 = buffersize - self.subprotocols["entangle_switch_" + str(i)].head - 1
                mem_pos_2 = buffersize - self.subprotocols["entangle_link_" + str(i)].head - 1
                if np.random.rand() > self.p[i]:
                    # Throw out qubit (Do this, because if I limit the generation itself probabilistically (via the
                    # QSource), then this protocol won't know when to stop yielding, since the EntangleNodes protocol
                    # would never stop waiting for a qubit to arrive).

                    # Update birthtimes to indicate these are now gone
                    network.get_node("switch_node_" + str(i)).subcomponents[
                        "QuantumMemory" + str(i)].qubit_birthtimes[mem_pos_1] = -1
                    network.get_node("link_node_" + str(i)).subcomponents[
                        "QuantumMemory" + str(i)].qubit_birthtimes[mem_pos_2] = -1

                    # Pop the qubits from the switch memories:
                    network.get_node("switch_node_" + str(i)).qmemory.pop(mem_pos_1)
                    network.get_node("link_node_" + str(i)).qmemory.pop(mem_pos_2)

                    self.subprotocols["entangle_switch_" + str(i)].head = (self.subprotocols["entangle_switch_" + str(
                        i)].head - 1) % buffersize
                    self.subprotocols["entangle_link_" + str(i)].head = (self.subprotocols["entangle_link_" + str(
                        i)].head - 1) % buffersize
                else:
                    # Track the cycle birthtime now that these qubits have survived
                    network.get_node("switch_node_" + str(i)).subcomponents[
                        "QuantumMemory" + str(i)].qubit_birthtimes_cycles[mem_pos_1] = self.nrounds - rounds_left
                    network.get_node("link_node_" + str(i)).subcomponents[
                        "QuantumMemory" + str(i)].qubit_birthtimes_cycles[mem_pos_2] = self.nrounds - rounds_left

            # Generate workload for this time step (use the "solution" variable for MW and Stationary):
            self.update_jobs(self.nrounds - rounds_left)
            # Job_queue is now updated and sorted according to scheduler

            # Iterate through job queue
            print("Jobs:", self.job_queue)
            # for job in range(0, len(self.job_queue)):
            job = 0
            while job < len(self.job_queue):
                # Get users from job queue
                user_1 = self.job_queue[job][0]
                user_2 = self.job_queue[job][1]
                if self.scheduler == "Stationary":
                    # LLE's are matched pre-emptively, read the destinations from the switch buffer and check if user_1
                    # and user_2 are matched

                    # (if user_1 isn't destined in a user_2 LLE or user_2 isn't destined in a user_1 LLE)...
                    if (user_1 not in network.get_node("switch_node_" + str(user_2)).subcomponents[
                        "QuantumMemory" + str(user_2)].destinations) or (user_2 not in network.get_node(
                            "switch_node_" + str(user_1)).subcomponents["QuantumMemory" + str(user_1)].destinations):
                        # ... then, go on to the next job to try again
                        job += 1
                        continue

                print("Trying to pair", user_1, user_2)
                # Python doesn't have do-while loops, so I'll do this instead :/
                while True:
                    # Update the fidelities:
                    self.update_fidelities()

                    # Get positions of qubits according to qubit policy
                    [pos_1, pos_2] = self.get_qubit_positions(user_1, user_2)

                    print(self.fidelities)
                    print("Using positions", pos_1, pos_2)

                    # This is the "while" part of the do-while loop
                    if pos_1 == -1 or pos_2 == -1:
                        # No qubits available for these users. Move to next job
                        print("No qubits available")
                        break

                    # Perform the entangling measurement
                    # Get switch qubits
                    q1 = network.get_node("switch_node_" + str(user_1)).qmemory.peek(pos_1)[0]
                    q2 = network.get_node("switch_node_" + str(user_2)).qmemory.peek(pos_2)[0]

                    # Run BSM
                    result, _ = self.BellStateMeasurement(q1, q2)

                    # Get link qubits
                    link_q1 = network.get_node("link_node_" + str(user_1)).qmemory.peek(pos_1)[0]
                    link_q2 = network.get_node("link_node_" + str(user_2)).qmemory.peek(pos_2)[0]

                    # Apply pauli operators
                    if result[0][0] == 1:
                        # Apply X to the qubit at the link node
                        network.get_node("link_node_" + str(user_2)).subcomponents[
                            "QuantumMemory" + str(user_2)].execute_instruction(instr.INSTR_X, [pos_2])
                        # Wait for processor if busy
                        if network.get_node("link_node_" + str(user_2)).qmemory.busy:
                            yield self.await_program(network.get_node("link_node_" + str(user_2)).qmemory)

                    if result[1][0] == 1:
                        # Apply Z to the qubit at the link node
                        network.get_node("link_node_" + str(user_2)).subcomponents[
                            "QuantumMemory" + str(user_2)].execute_instruction(instr.INSTR_Z, [pos_2])
                        # Wait for processor if busy
                        if network.get_node("link_node_" + str(user_2)).qmemory.busy:
                            yield self.await_program(network.get_node("link_node_" + str(user_2)).qmemory)

                    print("Fidelity of e2e:" + str(
                        ns.qubits.fidelity([link_q1, link_q2], ks.b00)))

                    # Update list of fidelities, throughput, and lifetimes
                    self.e2e_fidelities.append(ns.qubits.fidelity([link_q1, link_q2], ks.b00))
                    self.throughput[self.nrounds - rounds_left] += 1
                    # Lifetime = current round - birth round of qubit
                    self.lifetimes.append(
                        self.nrounds - rounds_left - network.get_node("switch_node_" + str(user_1)).subcomponents[
                            "QuantumMemory" + str(user_1)].qubit_birthtimes_cycles[pos_1])
                    self.lifetimes.append(
                        self.nrounds - rounds_left - network.get_node("switch_node_" + str(user_2)).subcomponents[
                            "QuantumMemory" + str(user_2)].qubit_birthtimes_cycles[pos_2])

                    # Update birthtimes to indicate these are now used
                    network.get_node("switch_node_" + str(user_1)).subcomponents[
                        "QuantumMemory" + str(user_1)].qubit_birthtimes[pos_1] = -1
                    network.get_node("switch_node_" + str(user_2)).subcomponents[
                        "QuantumMemory" + str(user_2)].qubit_birthtimes[pos_2] = -1

                    # Pop the qubits from the switch memories:
                    network.get_node("switch_node_" + str(user_1)).qmemory.pop(pos_1)
                    network.get_node("switch_node_" + str(user_2)).qmemory.pop(pos_2)
                    network.get_node("link_node_" + str(user_1)).qmemory.pop(pos_1)
                    network.get_node("link_node_" + str(user_2)).qmemory.pop(pos_2)

                    # Entanglement satisfied, update job queue
                    self.job_queue[job][2] -= 1

                    # Check if this is the first response to this job:
                    if not self.job_queue[job][4]:
                        # Update response time
                        self.job_queue[job][4] = True
                        self.response_times.append(self.nrounds - rounds_left - self.job_queue[job][5])

                    # Second condition
                    if self.job_queue[job][2] == 0:
                        # Job finished
                        print("Job done, moving on...")
                        # Update turnaround time
                        self.turnaround_times.append(self.nrounds - rounds_left - self.job_queue[job][5])
                        # Pop job off queue and update the indices so things don't go bad
                        self.job_queue.pop(job)
                        job -= 1
                        break

                job += 1

            # After the 'while job < len(self.job_queue)' loop
            # Update E[Q] data
            self.eq[self.nrounds - rounds_left] = sum(
                [network.get_node("switch_node_" + str(i)).qmemory.get_current_size()
                 for i in range(0, self.nlinks)])
            # Get total number of outstanding e2ee requests after this cycle
            self.outstanding_requests.append(sum([j[2] for j in self.job_queue]))
            rounds_left -= 1


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
        return size


class Plotter:
    def __init__(self, scheduler, qubit_scheduler, fidelities, throughput, eq, lifetimes, turnarounds, responses, outstanding_requests):
        plt.figure(figsize=(15, 10))
        plt.suptitle("" + str(len(eq)) + " cycles of " + scheduler + " with \"" + qubit_scheduler + "\" qubit policy on " + str(nlinks) + " links, buffer size of " + str(buffersize) + " per link, p=" + str(p), fontsize=16)
        # alpha for EWMA
        alpha = 0.05

        # Center confidence intervals at 0 to give a +-
        throughput_avg = np.average(throughput)
        throughput_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(throughput) / np.sqrt(len(throughput)))
        eq_avg = np.average(eq)
        eq_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(eq) / np.sqrt(len(eq)))

        # Get exponentially weighted moving average
        eq_exp = copy.deepcopy(eq)
        for i in range(1, len(eq_exp)):
            eq_exp[i] = (1 - alpha) * eq_exp[i - 1] + alpha * eq_exp[i]
        # Plot E[Q]
        plt.subplot(2, 3, 1)
        xrange = np.linspace(0, len(eq) - 1, len(eq))
        plt.plot(xrange, eq, label="# of qubits, avg=" + str(eq_avg) + "+-" + str(round(eq_conf[0], 3)))
        plt.plot(xrange, eq_exp, label="Moving average (EWMA)")
        plt.legend()
        plt.title("Memory Fullness vs Cycles")
        plt.ylabel("# of qubits total in switch memory")
        plt.xlabel("Cycle")

        # Get exponentially weighted moving average
        throughput_exp = copy.deepcopy(throughput)
        for i in range(1, len(throughput_exp)):
            throughput_exp[i] = (1 - alpha) * throughput_exp[i - 1] + alpha * throughput_exp[i]
        # Plot throughput
        plt.subplot(2, 3, 2)
        xrange = np.linspace(0, len(throughput) - 1, len(throughput))
        plt.plot(xrange, throughput, label="Throughput, avg=" + str(throughput_avg) + "+-" + str(round(throughput_conf[0], 3)))
        plt.plot(xrange, throughput_exp, label="Moving average (EWMA)")
        plt.plot(xrange, outstanding_requests, label="Outstanding Requests")
        plt.legend()
        plt.title("Throughput vs Cycles")
        plt.ylabel("E-2-E Entanglements")
        plt.xlabel("Cycle")

        # Plot fidelities
        plt.subplot(2, 3, 3)
        print(min(fidelities), max(fidelities))
        plt.hist(fidelities, bins=30, range=[0, 1])
        plt.title("Histogram of E-2-E Entanglement Fidelities")
        plt.ylabel("Count")
        plt.xlabel("Fidelity")

        # Plot lifetimes
        plt.subplot(2, 3, 4)
        plt.hist(lifetimes, bins=max(lifetimes))
        plt.title("Histogram of Qubit Lifetimes")
        plt.ylabel("Count")
        plt.xlabel("Lifetime (cycles)")

        # Plot turnaround times
        plt.subplot(2, 3, 5)
        plt.hist(turnarounds, bins=max(10, max(turnarounds)))
        plt.title("Histogram of Turnaround Times")
        plt.ylabel("Count")
        plt.xlabel("Turnaround Time (cycles)")

        # Plot response times
        plt.subplot(2, 3, 6)
        plt.hist(responses, bins=max(10, max(responses)))
        plt.title("Histogram of Response Times")
        plt.ylabel("Count")
        plt.xlabel("Response Time (cycles)")

        plt.show()


def network_setup(nlinks, buffersize, prep_delay=0):
    network = Network("Entangle_nodes")

    link_nodes = network.add_nodes(["link_node_" + str(i) for i in range(0, nlinks)])
    switch_nodes = network.add_nodes(["switch_node_" + str(i) for i in range(0, nlinks)])

    # Page 11: https://arxiv.org/pdf/1809.00364.pdf (?)
    # noise = T1T2NoiseModel(T1=1/3, T2=1/3)
    depolar = DepolarNoiseModel(time_independent=False, depolar_rate=1000000)
    dephase = DephaseNoiseModel(time_independent=False, dephase_rate=1000000)

    phys_instructions = [PhysicalInstruction(instr.INSTR_SWAP, duration=1, quantum_noise_model=dephase),
                         PhysicalInstruction(instr.INSTR_X, duration=1, quantum_noise_model=dephase),
                         PhysicalInstruction(instr.INSTR_Z, duration=1, quantum_noise_model=dephase)]

    # loss = FibreLossModel(p_loss_init=0.001, p_loss_length=0)
    fibre_delay = FibreDelayModel()
    fixed_delay = FixedDelayModel(delay=prep_delay)

    for i in range(0, nlinks):
        link_nodes[i].add_subcomponent(SwitchBuffer(
            "QuantumMemory" + str(i), buffersize, fallback_to_nonphysical=True,
            phys_instructions=phys_instructions,
            mem_noise_models=[depolar] * buffersize))
        switch_nodes[i].add_subcomponent(SwitchBuffer(
            "QuantumMemory" + str(i), buffersize, fallback_to_nonphysical=True,
            phys_instructions=phys_instructions,
            mem_noise_models=[depolar] * buffersize))

        link_nodes[i].add_subcomponent(
            QSource("QSource" + str(i), state_sampler=StateSampler(qreprs=[ks.b00]),
                    num_ports=2, status=SourceStatus.EXTERNAL,
                    models={"emission_delay_model": fixed_delay}))
        # Create and connect quantum channel:
        qchannel = QuantumChannel("QuantumChannel" + str(i), length=0.01,
                                  models={'quantum_noise_model': depolar,
                                          "delay_model": fibre_delay})
        port_name_a, port_name_b = network.add_connection(
            link_nodes[i], switch_nodes[i], channel_to=qchannel, label="connection" + str(i))
        # Setup Link ports:
        link_nodes[i].subcomponents["QSource" + str(i)].ports["qout0"].forward_output(
            link_nodes[i].ports[port_name_a])
        link_nodes[i].subcomponents["QSource" + str(i)].ports["qout1"].connect(
            link_nodes[i].qmemory.ports["qin0"])
        # Setup Switch ports:
        switch_nodes[i].ports[port_name_b].forward_input(switch_nodes[i].qmemory.ports["qin0"])
    return network


if __name__ == "__main__":
    ns.set_qstate_formalism(QFormalism.DM)
    ns.sim_reset()
    nlinks = 5
    buffersize = 50
    p = [0.9, 0.9, 0.9, 0.9, 0.9]

    network = network_setup(nlinks, buffersize)
    switch = SwitchProtocol(buffersize=buffersize, nlinks=nlinks, p=p, network=network, scheduler="Stationary",
                            workload="random",
                            rounds=1000, qubit_policy="oldest")
    switch.start()
    ns.sim_run()
