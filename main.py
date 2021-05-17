import os                                                                       
from multiprocessing import Pool
from time import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import copy
import sys

start = int(time() * 1000)
nruns = 12
nlinks = 5
buffersize = 100
scheduler = 'Stationary'
qubit_scheduler = 'oldest'
workload = 'Heavy'
p = 0.9
q = 0.9
nrounds = 4000
T_0 = ''


def run_process(params):
    _process, _nlinks, _buffersize, _scheduler, _p, _q, _nrounds, _qubit_scheduler, _workload, _T_0, _run_over, _T2 = params
    print(str(_process)+' '+str(_nlinks)+' '+str(_buffersize)+' '+str(_scheduler)+' '+str(_p)+' '+str(_q)+' '+
    str(_nrounds)+' '+str(_qubit_scheduler)+' '+str(_workload)+' '+str(_T_0)+' '+str(_run_over)+' '+str(_T2))
    os.system('/usr/bin/python3 \"/home/aarinaldi/NSSims/Switch Simulator.py\" '+
        str(_process)+' '+str(_nlinks)+' '+str(_buffersize)+' '+str(_scheduler)+' '+str(_p)+' '+str(_q)+' '+
        str(_nrounds)+' '+str(_qubit_scheduler)+' '+str(_workload)+' '+str(_T_0)+' '+str(_run_over)+' '+str(_T2))

def run_list(list):
    for job in list:
        print(job)
        tokens = job.split("_")
        if not os.path.exists('NSSims/Sim_Data/' + job):
            os.makedirs('NSSims/Sim_Data/' + job)
        else:
            print("ope")
            continue
        
        print(job)

        scheduler = tokens[1]
        buffersize = tokens[4] if scheduler == 'Max-Weight' else tokens[3]
        qubit_scheduler = 'oldest' if tokens[2] == 'O' else 'youngest'
        workload = tokens[0]
        p = tokens[5] if scheduler == 'Max-Weight' else tokens[4]
        q = tokens[6] if scheduler == 'Max-Weight' else tokens[5]
        T2 = tokens[7] if scheduler == 'Max-Weight' else tokens[6]
        run_over = tokens[8] if scheduler == 'Max-Weight' else tokens[7]
        nlinks = tokens[9] if scheduler == 'Max-Weight' else tokens[8]
        nrounds = 3000
        T_0 = tokens[3] if scheduler == 'Max-Weight' else ''

        print(nlinks)

        param_list = []
        for i in range(nruns):
            param_list.append((i, nlinks, buffersize, scheduler, p, q, nrounds, qubit_scheduler, workload, T_0, run_over, T2))
        pool = Pool(processes=nruns)
        print(param_list)
        pool.map(run_process, param_list)

        print("Finished simulation in "+str(int(time() * 1000) - start)+" ms")



# DATA PROCESSING

def just_print_values(to_print):
    for k in range(1, len(to_print)+1):
        print(to_print[k-1] + "...")
        tokens = to_print[k-1].split("_")
        
        workload = tokens[0]
        scheduler = tokens[1]
        qubit_scheduler = tokens[2]
        if scheduler == "Max-Weight":
            T_0 = tokens[3]
        
        eq = []
        throughput = []
        outstanding_requests = []
        response_times = []
        turnaround_times = []
        fidelities = []
        lifetimes = []
        for i in range(0, nruns):
            timestep_file = open('/home/aarinaldi/NSSims/Sim_Data/' + to_print[k-1] + '/timestep_data'+str(i)+'.npy', 'rb')
            job_file = open('/home/aarinaldi/NSSims/Sim_Data/' + to_print[k-1] + '/job_data'+str(i)+'.npy', 'rb')
            e2ee_file = open('/home/aarinaldi/NSSims/Sim_Data/' + to_print[k-1] + '/e2ee_data'+str(i)+'.npy', 'rb')

            timestep_temp = []
            while True:
                try:
                    timestep_temp.append(np.load(timestep_file, allow_pickle=True))
                except Exception as e:
                    break

            job_temp = []
            while True:
                try:
                    job_temp.append(np.load(job_file, allow_pickle=True))
                except Exception as e:
                    break

            e2ee_temp = []
            while True:
                try:
                    e2ee_temp.append(np.load(e2ee_file, allow_pickle=True))
                except Exception as e:
                    break

            timestep_data = np.stack(timestep_temp)
            job_data = np.stack(job_temp)
            e2ee_data = np.stack(e2ee_temp)

            eq.append(timestep_data[:,0])
            throughput.append(timestep_data[:,1])
            outstanding_requests.append(timestep_data[:,2])

            response_times.append(job_data[:,0])
            turnaround_times.append(job_data[:,1])

            fidelities.append(e2ee_data[:,0])
            lifetimes.append(np.append(e2ee_data[:,1], e2ee_data[:,2]))

        
        nrounds = len(eq[0])
        xrange = np.linspace(0, nrounds - 1, nrounds)

        string = ''
        if workload == 'Light':
            string = "30 percent capacity"
        elif workload == 'VeryHeavy':
            string = "98 percent capacity"
        else:
            string = "92 percent capacity"

        string2 = ''
        if scheduler == 'Max-Weight':
            string2 = "T_0 = "+str(T_0)

        if qubit_scheduler == "O":
            qubit_scheduler = "oldest"
        else:
            qubit_scheduler = "youngest"
            

        fidelities_full = np.array([])
        for i in range(0, nruns):
            fidelities_full = np.append(fidelities_full, np.array(fidelities[i]))

        fidelities_avg = np.average(fidelities_full)
        fidelities_avg_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(fidelities_full) / np.sqrt(len(fidelities_full)))

        print('\t Fidelities: ' + str(fidelities_avg) + ' +- ' + str(fidelities_avg_conf[1]))


        turnaround_times_full = np.array([])
        for i in range(0, nruns):
            turnaround_times_full = np.append(turnaround_times_full, np.array(turnaround_times[i]))

        turnaround_times_avg = np.average(turnaround_times_full)
        turnaround_times_avg_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(turnaround_times_full) / np.sqrt(len(turnaround_times_full)))

        print('\t Turnaround Times: ' + str(turnaround_times_avg) + ' +- ' + str(turnaround_times_avg_conf[1]))


        outstanding_requests_full = np.array([])
        for i in range(0, nruns):
            outstanding_requests_full = np.append(outstanding_requests_full, np.array(outstanding_requests[i][(nrounds//2):nrounds]))

        outstanding_requests_avg = np.average(outstanding_requests_full)
        outstanding_requests_avg_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(outstanding_requests_full) / np.sqrt(len(outstanding_requests_full)))

        print('\t Outstanding Requests: ' + str(outstanding_requests_avg) + ' +- ' + str(outstanding_requests_avg_conf[1]))
        

def plot_vs_q():
    plt.figure(figsize=(24, 6))
    outstanding_requests_final = [[],[],[],[]]
    turnaround_times_final = [[],[],[],[]]
    fidelities_final = [[],[],[],[]]

    outstanding_requests_final_err = [[],[],[],[]]
    turnaround_times_final_err = [[],[],[],[]]
    fidelities_final_err = [[],[],[],[]]

    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    params = [["O", "4"], ["O", "8"], ["Y", "4"], ["Y", "8"]]
    for j in range(0, len(params)):
        for q in qs:
            name = "Light_FIFO_"+params[j][0]+"_100_0.9_"+str(q)+"_1000000_q_"+params[j][1]
            print(name+ "...")
                        
            outstanding_requests = []
            turnaround_times = []
            fidelities = []
            for i in range(0, nruns):
                timestep_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/timestep_data'+str(i)+'.npy', 'rb')
                job_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/job_data'+str(i)+'.npy', 'rb')
                e2ee_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/e2ee_data'+str(i)+'.npy', 'rb')

                timestep_temp = []
                while True:
                    try:
                        timestep_temp.append(np.load(timestep_file, allow_pickle=True))
                    except Exception as e:
                        break

                job_temp = []
                while True:
                    try:
                        job_temp.append(np.load(job_file, allow_pickle=True))
                    except Exception as e:
                        break

                e2ee_temp = []
                while True:
                    try:
                        e2ee_temp.append(np.load(e2ee_file, allow_pickle=True))
                    except Exception as e:
                        break

                timestep_data = np.stack(timestep_temp)
                job_data = np.stack(job_temp)
                e2ee_data = np.stack(e2ee_temp)

                outstanding_requests.append(timestep_data[:,2])
                turnaround_times.append(job_data[:,1])
                fidelities.append(e2ee_data[:,0])

            nrounds = len(outstanding_requests[0]) 

            fidelities_full = np.array([])
            for i in range(0, nruns):
                fidelities_full = np.append(fidelities_full, np.array(fidelities[i]))

            fidelities_final[j].append(np.average(fidelities_full))
            fidelities_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(fidelities_full) / np.sqrt(len(fidelities_full)))[0])

            turnaround_times_full = np.array([])
            for i in range(0, nruns):
                turnaround_times_full = np.append(turnaround_times_full, np.array(turnaround_times[i]))

            turnaround_times_final[j].append(np.average(turnaround_times_full))
            turnaround_times_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(turnaround_times_full) / np.sqrt(len(turnaround_times_full)))[0])

            outstanding_requests_full = np.array([])
            for i in range(0, nruns):
                outstanding_requests_full = np.append(outstanding_requests_full, np.array(outstanding_requests[i][(nrounds//2):nrounds]))

            outstanding_requests_final[j].append(np.average(outstanding_requests_full))
            outstanding_requests_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(outstanding_requests_full) / np.sqrt(len(outstanding_requests_full)))[0])

            print("\tFidelity: "+str(np.average(fidelities_full)))
            print("\tTurnaround Time: "+str(np.average(turnaround_times_full)))
            print("\tOutstanding Requests: "+str(np.average(outstanding_requests_full)))

    print(fidelities_final)
    print(turnaround_times_final)
    print(outstanding_requests_final)
    


    # plt.subplot(1, 3, 1)
    # plt.errorbar(qs, fidelities_final[0], yerr=fidelities_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, fidelities_final[1], yerr=fidelities_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, fidelities_final[2], yerr=fidelities_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, fidelities_final[3], yerr=fidelities_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Fidelity vs q")
    # plt.ylabel("Fidelity")
    # plt.xlabel("q")

    # plt.subplot(1, 3, 2)
    # plt.errorbar(qs, turnaround_times_final[0], yerr=turnaround_times_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, turnaround_times_final[1], yerr=turnaround_times_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, turnaround_times_final[2], yerr=turnaround_times_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, turnaround_times_final[3], yerr=turnaround_times_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Turnaround Times vs T0")
    # plt.ylabel("Turnaround Time (ns)")
    # plt.xlabel("q")

    # plt.subplot(1, 3, 3)
    # plt.errorbar(qs, outstanding_requests_final[0], yerr=outstanding_requests_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, outstanding_requests_final[1], yerr=outstanding_requests_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, outstanding_requests_final[2], yerr=outstanding_requests_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, outstanding_requests_final[3], yerr=outstanding_requests_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Outstanding Requests vs T0")
    # plt.ylabel("E-2-E Entanglement Requests")
    # plt.xlabel("q")

    plt.suptitle(str(nruns) + " runs (per datapoint) of " + str(nrounds) + " cycles of" + str(scheduler))

    plt.tight_layout()
    plt.show()


def plot_vs_p():
    plt.figure(figsize=(24, 6))
    outstanding_requests_final = [[],[],[],[]]
    turnaround_times_final = [[],[],[],[]]
    fidelities_final = [[],[],[],[]]

    outstanding_requests_final_err = [[],[],[],[]]
    turnaround_times_final_err = [[],[],[],[]]
    fidelities_final_err = [[],[],[],[]]

    ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    params = [["O", "4"], ["O", "8"], ["Y", "4"], ["Y", "8"]]
    for j in range(0, len(params)):
        for p in ps:
            name = "Light_FIFO_"+params[j][0]+"_100_"+str(p)+"_0.9_1000000_p_"+params[j][1]
            print(name+ "...")
                        
            outstanding_requests = []
            turnaround_times = []
            fidelities = []
            for i in range(0, nruns):
                timestep_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/timestep_data'+str(i)+'.npy', 'rb')
                job_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/job_data'+str(i)+'.npy', 'rb')
                e2ee_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/e2ee_data'+str(i)+'.npy', 'rb')

                timestep_temp = []
                while True:
                    try:
                        timestep_temp.append(np.load(timestep_file, allow_pickle=True))
                    except Exception as e:
                        break

                job_temp = []
                while True:
                    try:
                        job_temp.append(np.load(job_file, allow_pickle=True))
                    except Exception as e:
                        break

                e2ee_temp = []
                while True:
                    try:
                        e2ee_temp.append(np.load(e2ee_file, allow_pickle=True))
                    except Exception as e:
                        break

                timestep_data = np.stack(timestep_temp)
                job_data = np.stack(job_temp)
                e2ee_data = np.stack(e2ee_temp)

                outstanding_requests.append(timestep_data[:,2])
                turnaround_times.append(job_data[:,1])
                fidelities.append(e2ee_data[:,0])

            nrounds = len(outstanding_requests[0]) 

            fidelities_full = np.array([])
            for i in range(0, nruns):
                fidelities_full = np.append(fidelities_full, np.array(fidelities[i]))

            fidelities_final[j].append(np.average(fidelities_full))
            fidelities_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(fidelities_full) / np.sqrt(len(fidelities_full)))[0])

            turnaround_times_full = np.array([])
            for i in range(0, nruns):
                turnaround_times_full = np.append(turnaround_times_full, np.array(turnaround_times[i]))

            turnaround_times_final[j].append(np.average(turnaround_times_full))
            turnaround_times_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(turnaround_times_full) / np.sqrt(len(turnaround_times_full)))[0])

            outstanding_requests_full = np.array([])
            for i in range(0, nruns):
                outstanding_requests_full = np.append(outstanding_requests_full, np.array(outstanding_requests[i][(nrounds//2):nrounds]))

            outstanding_requests_final[j].append(np.average(outstanding_requests_full))
            outstanding_requests_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(outstanding_requests_full) / np.sqrt(len(outstanding_requests_full)))[0])

            print("\tFidelity: "+str(np.average(fidelities_full)))
            print("\tTurnaround Time: "+str(np.average(turnaround_times_full)))
            print("\tOutstanding Requests: "+str(np.average(outstanding_requests_full)))

    print(fidelities_final)
    print(turnaround_times_final)
    print(outstanding_requests_final)
    


    # plt.subplot(1, 3, 1)
    # plt.errorbar(qs, fidelities_final[0], yerr=fidelities_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, fidelities_final[1], yerr=fidelities_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, fidelities_final[2], yerr=fidelities_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, fidelities_final[3], yerr=fidelities_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Fidelity vs q")
    # plt.ylabel("Fidelity")
    # plt.xlabel("q")

    # plt.subplot(1, 3, 2)
    # plt.errorbar(qs, turnaround_times_final[0], yerr=turnaround_times_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, turnaround_times_final[1], yerr=turnaround_times_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, turnaround_times_final[2], yerr=turnaround_times_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, turnaround_times_final[3], yerr=turnaround_times_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Turnaround Times vs T0")
    # plt.ylabel("Turnaround Time (ns)")
    # plt.xlabel("q")

    # plt.subplot(1, 3, 3)
    # plt.errorbar(qs, outstanding_requests_final[0], yerr=outstanding_requests_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, outstanding_requests_final[1], yerr=outstanding_requests_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, outstanding_requests_final[2], yerr=outstanding_requests_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, outstanding_requests_final[3], yerr=outstanding_requests_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Outstanding Requests vs T0")
    # plt.ylabel("E-2-E Entanglement Requests")
    # plt.xlabel("q")

    plt.suptitle(str(nruns) + " runs (per datapoint) of " + str(nrounds) + " cycles of" + str(scheduler))

    plt.tight_layout()
    plt.show()


def plot_vs_B():
    plt.figure(figsize=(24, 6))
    outstanding_requests_final = [[],[],[],[]]
    turnaround_times_final = [[],[],[],[]]
    fidelities_final = [[],[],[],[]]

    outstanding_requests_final_err = [[],[],[],[]]
    turnaround_times_final_err = [[],[],[],[]]
    fidelities_final_err = [[],[],[],[]]

    Bs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    params = [["O", "4"], ["O", "8"], ["Y", "4"], ["Y", "8"]]
    for j in range(0, len(params)):
        for B in Bs:
            name = "VeryHeavy_FIFO_"+params[j][0]+"_"+str(B)+"_0.9_0.9_1000000_B_"+params[j][1]
            print(name+ "...")
                        
            outstanding_requests = []
            turnaround_times = []
            fidelities = []
            for i in range(0, nruns):
                timestep_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/timestep_data'+str(i)+'.npy', 'rb')
                job_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/job_data'+str(i)+'.npy', 'rb')
                e2ee_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/e2ee_data'+str(i)+'.npy', 'rb')

                timestep_temp = []
                while True:
                    try:
                        timestep_temp.append(np.load(timestep_file, allow_pickle=True))
                    except Exception as e:
                        break

                job_temp = []
                while True:
                    try:
                        job_temp.append(np.load(job_file, allow_pickle=True))
                    except Exception as e:
                        break

                e2ee_temp = []
                while True:
                    try:
                        e2ee_temp.append(np.load(e2ee_file, allow_pickle=True))
                    except Exception as e:
                        break

                timestep_data = np.stack(timestep_temp)
                job_data = np.stack(job_temp)
                e2ee_data = np.stack(e2ee_temp)

                outstanding_requests.append(timestep_data[:,2])
                turnaround_times.append(job_data[:,1])
                fidelities.append(e2ee_data[:,0])

            nrounds = len(outstanding_requests[0]) 

            fidelities_full = np.array([])
            for i in range(0, nruns):
                fidelities_full = np.append(fidelities_full, np.array(fidelities[i]))

            fidelities_final[j].append(np.average(fidelities_full))
            fidelities_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(fidelities_full) / np.sqrt(len(fidelities_full)))[0])

            turnaround_times_full = np.array([])
            for i in range(0, nruns):
                turnaround_times_full = np.append(turnaround_times_full, np.array(turnaround_times[i]))

            turnaround_times_final[j].append(np.average(turnaround_times_full))
            turnaround_times_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(turnaround_times_full) / np.sqrt(len(turnaround_times_full)))[0])

            outstanding_requests_full = np.array([])
            for i in range(0, nruns):
                outstanding_requests_full = np.append(outstanding_requests_full, np.array(outstanding_requests[i][(nrounds//2):nrounds]))

            outstanding_requests_final[j].append(np.average(outstanding_requests_full))
            outstanding_requests_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(outstanding_requests_full) / np.sqrt(len(outstanding_requests_full)))[0])

            print("\tFidelity: "+str(np.average(fidelities_full)))
            print("\tTurnaround Time: "+str(np.average(turnaround_times_full)))
            print("\tOutstanding Requests: "+str(np.average(outstanding_requests_full)))

    print(fidelities_final)
    print(turnaround_times_final)
    print(outstanding_requests_final)
    


    # plt.subplot(1, 3, 1)
    # plt.errorbar(qs, fidelities_final[0], yerr=fidelities_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, fidelities_final[1], yerr=fidelities_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, fidelities_final[2], yerr=fidelities_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, fidelities_final[3], yerr=fidelities_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Fidelity vs q")
    # plt.ylabel("Fidelity")
    # plt.xlabel("q")

    # plt.subplot(1, 3, 2)
    # plt.errorbar(qs, turnaround_times_final[0], yerr=turnaround_times_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, turnaround_times_final[1], yerr=turnaround_times_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, turnaround_times_final[2], yerr=turnaround_times_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, turnaround_times_final[3], yerr=turnaround_times_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Turnaround Times vs T0")
    # plt.ylabel("Turnaround Time (ns)")
    # plt.xlabel("q")

    # plt.subplot(1, 3, 3)
    # plt.errorbar(qs, outstanding_requests_final[0], yerr=outstanding_requests_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, outstanding_requests_final[1], yerr=outstanding_requests_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, outstanding_requests_final[2], yerr=outstanding_requests_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, outstanding_requests_final[3], yerr=outstanding_requests_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Outstanding Requests vs T0")
    # plt.ylabel("E-2-E Entanglement Requests")
    # plt.xlabel("q")

    plt.suptitle(str(nruns) + " runs (per datapoint) of " + str(nrounds) + " cycles of" + str(scheduler))

    plt.tight_layout()
    plt.show()


def plot_vs_T2():
    plt.figure(figsize=(24, 6))
    outstanding_requests_final = [[],[],[],[]]
    turnaround_times_final = [[],[],[],[]]
    fidelities_final = [[],[],[],[]]

    outstanding_requests_final_err = [[],[],[],[]]
    turnaround_times_final_err = [[],[],[],[]]
    fidelities_final_err = [[],[],[],[]]

    t2s = [200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000]
    params = [["O", "4"], ["O", "8"], ["Y", "4"], ["Y", "8"]]
    for j in range(0, len(params)):
        for t2 in t2s:
            name = "VeryHeavy_FIFO_"+params[j][0]+"_100_0.9_0.9_"+str(t2)+"_T2_"+params[j][1]
            print(name+ "...")
                        
            outstanding_requests = []
            turnaround_times = []
            fidelities = []
            for i in range(0, nruns):
                timestep_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/timestep_data'+str(i)+'.npy', 'rb')
                job_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/job_data'+str(i)+'.npy', 'rb')
                e2ee_file = open('/home/aarinaldi/NSSims/Sim_Data/' + name + '/e2ee_data'+str(i)+'.npy', 'rb')

                timestep_temp = []
                while True:
                    try:
                        timestep_temp.append(np.load(timestep_file, allow_pickle=True))
                    except Exception as e:
                        break

                job_temp = []
                while True:
                    try:
                        job_temp.append(np.load(job_file, allow_pickle=True))
                    except Exception as e:
                        break

                e2ee_temp = []
                while True:
                    try:
                        e2ee_temp.append(np.load(e2ee_file, allow_pickle=True))
                    except Exception as e:
                        break

                timestep_data = np.stack(timestep_temp)
                job_data = np.stack(job_temp)
                e2ee_data = np.stack(e2ee_temp)

                outstanding_requests.append(timestep_data[:,2])
                turnaround_times.append(job_data[:,1])
                fidelities.append(e2ee_data[:,0])

            nrounds = len(outstanding_requests[0]) 

            fidelities_full = np.array([])
            for i in range(0, nruns):
                fidelities_full = np.append(fidelities_full, np.array(fidelities[i]))

            fidelities_final[j].append(np.average(fidelities_full))
            fidelities_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(fidelities_full) / np.sqrt(len(fidelities_full)))[0])

            turnaround_times_full = np.array([])
            for i in range(0, nruns):
                turnaround_times_full = np.append(turnaround_times_full, np.array(turnaround_times[i]))

            turnaround_times_final[j].append(np.average(turnaround_times_full))
            turnaround_times_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(turnaround_times_full) / np.sqrt(len(turnaround_times_full)))[0])

            outstanding_requests_full = np.array([])
            for i in range(0, nruns):
                outstanding_requests_full = np.append(outstanding_requests_full, np.array(outstanding_requests[i][(nrounds//2):nrounds]))

            outstanding_requests_final[j].append(np.average(outstanding_requests_full))
            outstanding_requests_final_err[j].append(scipy.stats.norm.interval(0.95, loc=0, scale=np.std(outstanding_requests_full) / np.sqrt(len(outstanding_requests_full)))[0])

            print("\tFidelity: "+str(np.average(fidelities_full)))
            print("\tTurnaround Time: "+str(np.average(turnaround_times_full)))
            print("\tOutstanding Requests: "+str(np.average(outstanding_requests_full)))

    print(fidelities_final)
    print(turnaround_times_final)
    print(outstanding_requests_final)
    


    # plt.subplot(1, 3, 1)
    # plt.errorbar(qs, fidelities_final[0], yerr=fidelities_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, fidelities_final[1], yerr=fidelities_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, fidelities_final[2], yerr=fidelities_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, fidelities_final[3], yerr=fidelities_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Fidelity vs q")
    # plt.ylabel("Fidelity")
    # plt.xlabel("q")

    # plt.subplot(1, 3, 2)
    # plt.errorbar(qs, turnaround_times_final[0], yerr=turnaround_times_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, turnaround_times_final[1], yerr=turnaround_times_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, turnaround_times_final[2], yerr=turnaround_times_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, turnaround_times_final[3], yerr=turnaround_times_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Turnaround Times vs T0")
    # plt.ylabel("Turnaround Time (ns)")
    # plt.xlabel("q")

    # plt.subplot(1, 3, 3)
    # plt.errorbar(qs, outstanding_requests_final[0], yerr=outstanding_requests_final_err[0], label="Oldest, K=4")
    # plt.errorbar(qs, outstanding_requests_final[1], yerr=outstanding_requests_final_err[1], label="Oldest, K=8")
    # plt.errorbar(qs, outstanding_requests_final[2], yerr=outstanding_requests_final_err[2], label="Youngest, K=4")
    # plt.errorbar(qs, outstanding_requests_final[3], yerr=outstanding_requests_final_err[3], label="Youngest, K=8")
    # plt.legend()
    # plt.title("Outstanding Requests vs T0")
    # plt.ylabel("E-2-E Entanglement Requests")
    # plt.xlabel("q")

    plt.suptitle(str(nruns) + " runs (per datapoint) of " + str(nrounds) + " cycles of" + str(scheduler))

    plt.tight_layout()
    plt.show()


def plot_all_in_list(to_plot):
    plt.figure(figsize=(35, 5*len(to_plot)))
    for k in range(1, len(to_plot)+1):
        print(to_plot[k-1] + "...")
        tokens = to_plot[k-1].split("_")
        
        workload = tokens[0]
        scheduler = tokens[1]
        qubit_scheduler = tokens[2]
        if scheduler == "Max-Weight":
            T_0 = tokens[3]
        
        eq = []
        throughput = []
        outstanding_requests = []
        response_times = []
        turnaround_times = []
        fidelities = []
        lifetimes = []
        for i in range(0, nruns):
            timestep_file = open('/home/aarinaldi/NSSims/Sim_Data/' + to_plot[k-1] + '/timestep_data'+str(i)+'.npy', 'rb')
            job_file = open('/home/aarinaldi/NSSims/Sim_Data/' + to_plot[k-1] + '/job_data'+str(i)+'.npy', 'rb')
            e2ee_file = open('/home/aarinaldi/NSSims/Sim_Data/' + to_plot[k-1] + '/e2ee_data'+str(i)+'.npy', 'rb')

            timestep_temp = []
            while True:
                try:
                    timestep_temp.append(np.load(timestep_file, allow_pickle=True))
                except Exception as e:
                    break

            job_temp = []
            while True:
                try:
                    job_temp.append(np.load(job_file, allow_pickle=True))
                except Exception as e:
                    break

            e2ee_temp = []
            while True:
                try:
                    e2ee_temp.append(np.load(e2ee_file, allow_pickle=True))
                except Exception as e:
                    break

            timestep_data = np.stack(timestep_temp)
            job_data = np.stack(job_temp)
            e2ee_data = np.stack(e2ee_temp)

            eq.append(timestep_data[:,0])
            throughput.append(timestep_data[:,1])
            outstanding_requests.append(timestep_data[:,2])

            response_times.append(job_data[:,0])
            turnaround_times.append(job_data[:,1])

            fidelities.append(e2ee_data[:,0])
            lifetimes.append(np.append(e2ee_data[:,1], e2ee_data[:,2]))

        
        nrounds = len(eq[0])
        xrange = np.linspace(0, nrounds - 1, nrounds)

        string = ''
        if workload == 'Light':
            string = "30 percent capacity"
        elif workload == 'VeryHeavy':
            string = "98 percent capacity"
        else:
            string = "92 percent capacity"

        string2 = ''
        if scheduler == 'Max-Weight':
            string2 = "T_0 = "+str(T_0)

        if qubit_scheduler == "O":
            qubit_scheduler = "oldest"
        else:
            qubit_scheduler = "youngest"
            


        # eq_full = np.array([])
        # for i in range(0, nruns):
        #     eq_full = np.append(eq_full, np.array(eq[i][(nrounds//2):nrounds]))

        # eq_avg = np.average(eq_full)
        # eq_avg_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(eq_full) / np.sqrt(len(eq_full)))
        # eq_r = []

        # # Plot eq
        # plt.subplot(len(to_plot), 6, 6*(k-1) + 1)
        # for i in range(0, nruns):
        #     plt.plot(xrange[0:len(eq[i])], eq[i])
            
        # for i in range(0, nrounds):
        #     eq_r.append(np.average([ eq[j][i] for j in range(nruns) ]))

        # plt.plot(xrange[0:len(eq[0])], eq_r, 'b-')
        # plt.title("Buffer Occupancy vs Time\n Avg = "+str(round(eq_avg, 3))+" +- "+str(round(eq_avg_conf[0], 3)))
        # plt.ylabel("# of qubits in buffer")
        # plt.xlabel("Time (cycles)")
        # plt.text(-4000*(nrounds/3000), 100, str(scheduler) + "\n" + str(nruns) + " runs of " + str(nrounds) + " cycles" + 
        #     "\n\"" + qubit_scheduler + "\" qubit policy \n" + 
        #     "NLinks = " + str(nlinks) + "\nBuffer size = " + str(buffersize) + " per link\n p=" + str(p) + 
        #     "\n q = " + str(q) + " \n" + string + "\n" + string2, fontsize=20)



        # throughput_full = np.array([])
        # for i in range(0, nruns):
        #     throughput_full = np.append(throughput_full, np.array(throughput[i][(nrounds//2):nrounds]))

        # throughput_avg = np.average(throughput_full)
        # throughput_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(throughput_full) / np.sqrt(len(throughput_full)))

        # # Plot throughput
        # plt.subplot(len(to_plot), 6, 6*(k-1) + 2)
        # for i in range(0, nruns):
        #     plt.plot(xrange[0:len(throughput[i])], throughput[i])
        # plt.title("Throughput vs Cycles\n Avg = "+str(round(throughput_avg, 3))+" +- "+str(round(throughput_conf[0], 3)))
        # plt.ylabel("E-2-E Entanglements")
        # plt.xlabel("Cycle")



        fidelities_full = np.array([])
        for i in range(0, nruns):
            fidelities_full = np.append(fidelities_full, np.array(fidelities[i]))

        fidelities_avg = np.average(fidelities_full)
        fidelities_avg_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(fidelities_full) / np.sqrt(len(fidelities_full)))

        # Plot fidelities
        plt.subplot(len(to_plot), 3, 6*(k-1) + 3)
        plt.hist(fidelities_full, bins=30, range=[0, 1])
        plt.title("E-2-E Entanglement Fidelities\n Avg = "+str(round(fidelities_avg, 3))+" +- "+str(round(fidelities_avg_conf[0], 3)))
        plt.ylabel("Count")
        plt.xlabel("Fidelity")



      
        # lifetimes_full = np.array([])
        # for i in range(0, nruns):
        #     lifetimes_full = np.append(lifetimes_full, np.array(lifetimes[i]))

        # lifetimes_avg = np.average(lifetimes_full)
        # lifetimes_avg_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(lifetimes_full) / np.sqrt(len(lifetimes_full)))

        # # Plot lifetimes
        # plt.subplot(len(to_plot), 6, 6*(k-1) + 4)
        # plt.hist(lifetimes_full, bins=30, range=[0, max(lifetimes_full)])
        # plt.title("Qubit lifetimes\n Avg = "+str(round(lifetimes_avg, 3))+" +- "+str(round(lifetimes_avg_conf[0], 3)))
        # plt.ylabel("Count")
        # plt.xlabel("Lifetime (ns)")

        turnaround_times_full = np.array([])
        for i in range(0, nruns):
            turnaround_times_full = np.append(turnaround_times_full, np.array(turnaround_times[i]))

        turnaround_times_avg = np.average(turnaround_times_full)
        turnaround_times_avg_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(turnaround_times_full) / np.sqrt(len(turnaround_times_full)))


        turnaround_times_full = np.array([])
        for i in range(0, nruns):
            turnaround_times_full = np.append(turnaround_times_full, np.array(turnaround_times[i]))

        turnaround_times_avg = np.average(turnaround_times_full)
        turnaround_times_avg_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(turnaround_times_full) / np.sqrt(len(turnaround_times_full)))

        # Plot turnaround times
        plt.subplot(len(to_plot), 3, 6*(k-1) + 5)
        plt.hist(turnaround_times_full, bins=20)
        plt.title("Histogram of Turnaround Times\n Avg = "+str(round(turnaround_times_avg, 3))+" +- "+str(round(turnaround_times_avg_conf[0], 3)))
        plt.ylabel("Count")
        plt.xlabel("Turnaround Time (ns)")



        outstanding_requests_full = np.array([])
        for i in range(0, nruns):
            outstanding_requests_full = np.append(outstanding_requests_full, np.array(outstanding_requests[i][(nrounds//2):nrounds]))

        outstanding_requests_avg = np.average(outstanding_requests_full)
        outstanding_requests_avg_conf = scipy.stats.norm.interval(0.95, loc=0, scale=np.std(outstanding_requests_full) / np.sqrt(len(outstanding_requests_full)))
        outstanding_requests_r = []

        # Plot outstanding requests
        plt.subplot(len(to_plot), 3, 6*(k-1) + 6)
        for i in range(0, nruns):
            plt.plot(xrange[0:len(throughput[i])], outstanding_requests[i])
            
        for i in range(0, nrounds):
            outstanding_requests_r.append(np.average([ outstanding_requests[j][i] for j in range(nruns) ]))

        plt.plot(xrange[0:len(throughput[0])], outstanding_requests_r, 'b-')
        plt.title("Outstanding Requests vs Cycles\n Avg = "+str(round(outstanding_requests_avg, 3))+" +- "+str(round(outstanding_requests_avg_conf[0], 3)))
        plt.ylabel("# of requests in queue")
        plt.xlabel("Time (cycles)")


    plt.tight_layout()
    plt.show()



# l = ["Heavy_Stationary_O", "Light_FIFO_O", "Light_FIFO_Y", "Light_Max-Weight_O_1", "Light_Max-Weight_O_20", 
#         "Light_Max-Weight_Y_1", "Light_Max-Weight_Y_20", "Light_Stationary_O", "Light_Stationary_Y", 
#         "VeryHeavy_FIFO_O", "VeryHeavy_FIFO_Y", "VeryHeavy_Max-Weight_O_1", "VeryHeavy_Max-Weight_O_20", 
#         "VeryHeavy_Max-Weight_Y_1", "VeryHeavy_Max-Weight_Y_20", "VeryHeavy_Stationary_O", "VeryHeavy_Stationary_Y"]

# l = ["VeryHeavy_FIFO_O_100_0.9_0.9_1000000_B"]
l = []
for k in [4, 8]:
    for s in ["O", "Y"]:
        for i in [1.0]:
            l.append("Light_FIFO_"+s+"_100_0.9_"+str(i)+"_1000000_q_"+str(k))
        for i in [1.0]:
            l.append("Light_FIFO_"+s+"_100_"+str(i)+"_0.9_1000000_p_"+str(k))
        for T in [200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000]:
            l.append("VeryHeavy_FIFO_"+s+"_100_0.9_0.9_"+str(T)+"_T2_"+str(k))
        
# for k in [4, 8]:
#     for s in ["O", "Y"]:
#         for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#             l.append("Light_FIFO_"+s+"_100_0.9_"+str(i)+"_1000000_q_"+str(k))
#         for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#             l.append("Light_FIFO_"+s+"_100_"+str(i)+"_0.9_1000000_p_"+str(k))
#         for b in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
#             l.append("VeryHeavy_FIFO_"+s+"_"+str(b)+"_0.9_0.9_1000000_B_"+str(k))
#         for T in [200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000]:
#             l.append("VeryHeavy_FIFO_"+s+"_100_0.9_0.9_"+str(T)+"_B_"+str(k))

# print(l)
# run_list(l)
plot_vs_B()
# just_print_values(l)
