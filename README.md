# NetSquid-Simulations
Quantum Network Simulations


--------------------------------------------
Nodes:

    Link Nodes:
        TODO
    Switch Nodes:
        TODO
    
--------------------------------------------
Protocols:

    EntangleNode:
    -Used for generating link level entanglements
    -N instances run for N users
    -member variables
        _is_source: True if node running the protocol is link node
        start_expression: EventExpression used to signal start of new cycle
        _num_pairs: number of paris used for multiplexing (default=1)
        _mem_positions: claimed memory positions to be used in protocol
        _input_mem_pos: memory position that will recieve the qubit from the QSource
        _qmem_input_port: port that connects the channel from QSource to the node
        haed: current position of cyclic memory
    -functions:
        __init__()
            initialize parameters and check validity of memory
        start()
            start protocol (claim memory and set the entangled_pairs counter to 0
        stop()
            stop protocol (unclaim memory spaces)
        run()
            Each cycle go to next memory position. New qubits are inputted into the input pos, so spawn the inpus position with the head position
                -INSTR_SWAP does this
                
    SwitchProtocol:
    -Main controller of the program
    -Signals for new entanglement generation
    -member variables:
        scheduler: String signifying what scheduler to user
            -"FIFO": Job priority is determined by job arrival time
            -"SJF": Job priority is determined by total job size, increasing order
            -"STCF": Job priority is determined by how many e2e entanglements are left in job, increasing order
            -"LJF": Job priority is determined by total job size, decreasing order
            -"LTCF": Job priority is determined by how many e2e entanglements are left in job, decreasing order
            -"OQF": Job priority is determined by how old the qubits are in the link's memories, decreasing order
            -"YQF": Job priority is determined by how old the qubits are in the link's memories, increasing order
        nrounds: Total number of entanglement rounds to run the simulation for
        nlinks: Number of links attached to switch
        buffersize: size of each of the switch memories
        workload: String signifying what workload will be running
            -"random": bins entanglement requests into a random amount of jobs of random sizes between random users
        qubit_policy: How qubits are chosen from memory to be used in e2e entanglements
            -"oldest": Always choose the oldest qubits in memory
            -"youngest": Always choose the youngest qubits in memory
        network: Object holding all the information about the network
        fidelities: Matrix of floats giving the fidelities of all qubits in all switch memories
        job_queue: Queue (Python List) of jobs 
    -data collection variables:
        e2e_fidelities: list of fidelities to be graphed
        eq: array of how full the memories are in each cycle
        throughput: array of how many e2e-entanglements are generated in each time slot
        lifetimes: list of qubit lifetimes in memory (only qubits that are USED in e2e entanglements)
        turnaround_times: list of turnaround times of jobs (time_satisfied - time_arrival)
        response_times: listof response times of jobs (time_first_responded_to - time_arrival)
    -functions:
        __init__()
            initialize the variables above, and set the subprotocols
        gen_kron()
            useful function for multiplying an array of matrices together. Made by Yuan and ERic
        depolarization()
            custom depolarizaiton model for the bell state measurement, made by Yuan and Eric
        BellStateMeasurement()
            Apply measurment noise to qubits, run a BSM, and return the results (ie: how many X and Z gates to apply)
        update_jobs()
            Runs once each cycle. Adds jobs to self.job_queue according to what workload is specified, and orders according to the scheduler
        update_fidelities()
            updates self.fidelities 
        run()
            Wait for the link entanglements, and then call the update_jobs function. Using this info, perform the BSM's
                and apply the approriate Pauli operations, and finally update the data variables
