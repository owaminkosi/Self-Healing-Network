import networkx as nx
import random
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd

# =====================================================
# GLOBALS
# =====================================================

Q_table = {}
actions = ["reroute", "reduce_load", "do_nothing"]

metrics_log = []

# =====================================================
# NETWORK CREATION
# =====================================================

def create_dual_star(num_nodes=20):
    G = nx.Graph()

    # Add nodes
    for i in range(1, num_nodes+1):
        G.add_node(i, status="active")

    primary = 1
    backup = 2

    for node in range(3, num_nodes+1):
        G.add_edge(primary, node, load=0)
        G.add_edge(backup, node, load=0)

    G.add_edge(primary, backup, load=0)

    return G

# =====================================================
# CASCADING FAILURE SIMULATION
# =====================================================

def cascading_failure(G, threshold=15):
    failed_nodes = []

    for e in list(G.edges()):
        if G.edges[e]["load"] > threshold:
            G.remove_edge(*e)

            # Chance of node failure
            if random.random() < 0.3:
                node_to_fail = random.choice(e)
                G.nodes[node_to_fail]["status"] = "failed"
                failed_nodes.append(node_to_fail)

    if failed_nodes:
        print("Cascading failures:", failed_nodes)



# =====================================================
# PACKET FLOW SIMULATION
# =====================================================

def simulate_packets(G, num_packets=20):
    sent = num_packets
    delivered = 0
    total_delay = 0

    active_nodes = [n for n,d in G.nodes(data=True) if d["status"]=="active"]

    for _ in range(num_packets):
        if len(active_nodes) < 2:
            break

        src, dst = random.sample(active_nodes, 2)

        try:
            path = nx.shortest_path(G, src, dst)
            delay = sum(G.edges[(path[i], path[i+1])]["load"]+1
                        for i in range(len(path)-1))
            delivered += 1
            total_delay += delay
        except:
            pass

    loss = sent - delivered
    avg_delay = total_delay/delivered if delivered else 0

    return sent, delivered, loss, avg_delay


# =====================================================
# ATTACK SIMULATION (DoS)
# =====================================================

def dos_attack(G, target=1):
    print("DoS attack on hub", target)
    for neighbor in list(G.neighbors(target)):
        G.edges[(target, neighbor)]["load"] += 15


# =====================================================
# CONGESTION UPDATE
# =====================================================

def random_congestion(G):
    for e in G.edges():
        G.edges[e]["load"] += random.randint(0,3)


def reduce_congestion(G):
    for e in G.edges():
        G.edges[e]["load"] = max(0, G.edges[e]["load"] - 5)


# =====================================================
# RL LOGIC
# =====================================================

def get_state(G):
    connected = nx.is_connected(G)
    congestion = any(G.edges[e]["load"] > 10 for e in G.edges())
    return (connected, congestion)


def choose_action(state, mode="RL"):
    if mode == "traditional":
        return "reroute"

    if random.random() < 0.2:
        return random.choice(actions)

    if state not in Q_table:
        Q_table[state] = {a:0 for a in actions}

    return max(Q_table[state], key=Q_table[state].get)


def update_q(state, action, reward):
    alpha = 0.1
    gamma = 0.9

    if state not in Q_table:
        Q_table[state] = {a:0 for a in actions}

    current_q = Q_table[state][action]
    Q_table[state][action] = current_q + alpha * (
        reward + gamma * max(Q_table[state].values()) - current_q
    )


def apply_action(G, action):
    reward = 0

    if action == "reduce_load":
        reduce_congestion(G)
        reward += 10

    elif action == "reroute":
        reward += 5

    elif action == "do_nothing":
        reward -= 5

    return reward


# =====================================================
# MAIN SIMULATION
# =====================================================

def run_simulation(mode="RL", steps=20, num_nodes=20):

    G = create_dual_star(num_nodes)
    controller = SDNController(mode)

    recovery_start = None
    recovery_times = []

    for t in range(steps):

        random_congestion(G)
        cascading_failure(G)

        # DoS attack at step 5
        if t == 5:
            dos_attack(G)

        # ------------------------------
        # SDN CONTROLLER LOGIC
        # ------------------------------
        state = controller.monitor(G)
        action = controller.decide(state)
        reward = controller.enforce(G, action)

        if mode == "RL":
            update_q(state, action, reward)

        # ------------------------------
        # RECOVERY TIME TRACKING
        # ------------------------------
        if not state[0] and recovery_start is None:
            recovery_start = t

        if state[0] and recovery_start is not None:
            recovery_times.append(t - recovery_start)
            recovery_start = None

        sent, delivered, loss, delay = simulate_packets(G)

        metrics_log.append([
            mode, t, sent, delivered, loss, delay, state[0]
        ])

    mttr = sum(recovery_times)/len(recovery_times) if recovery_times else 0
    return mttr



# =====================================================
# AUTOMATTIC PERFORMANCE GRAPHS
# =====================================================

def plot_results():
    df = pd.read_csv("simulation_results.csv")

    for metric in ["Packet_Loss", "Avg_Delay"]:
        plt.figure()
        for mode in df["Mode"].unique():
            subset = df[df["Mode"] == mode]
            plt.plot(subset["Time"], subset[metric], label=mode)

        plt.title(metric + " Over Time")
        plt.xlabel("Time")
        plt.ylabel(metric)
        plt.legend()
        plt.show()

# =====================================================
# STATISTICAL TESTING (t-test)
# =====================================================

        from scipy.stats import ttest_ind

def statistical_test():
    df = pd.read_csv("simulation_results.csv")

    rl_loss = df[df["Mode"]=="RL"]["Packet_Loss"]
    trad_loss = df[df["Mode"]=="traditional"]["Packet_Loss"]

    stat, p = ttest_ind(rl_loss, trad_loss)

    print("T-test statistic:", stat)
    print("P-value:", p)

    if p < 0.05:
        print("Statistically significant difference detected.")
    else:
        print("No statistically significant difference.")

# =====================================================
# SDN CONTROLLER CLASS
# =====================================================

class SDNController:

    def __init__(self, mode="RL"):   # ← add mode here
        self.mode = mode   # store mode ("RL" or "traditional")

    def monitor(self, G):
        return get_state(G)

    def decide(self, state):
        return choose_action(state, self.mode)

    def enforce(self, G, action):
        return apply_action(G, action)


# =====================================================
# EXPORT CSV
# =====================================================

def export_results():
    with open("simulation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Mode", "Time", "Packets_Sent",
            "Packets_Delivered", "Packet_Loss",
            "Avg_Delay", "Connected"
        ])
        writer.writerows(metrics_log)


# =====================================================
# RUN BOTH MODES
# =====================================================

if __name__ == "__main__":

    print("Running Traditional Routing...")
    
    mttr_traditional = run_simulation(mode="traditional")
    mttr_rl = run_simulation(mode="RL")


    print("Running Reinforcement Learning...")
    mttr_rl = run_simulation(mode="RL")

    export_results()

    print("\nMean Time To Recovery:")
    print("Traditional:", mttr_traditional)
    print("RL:", mttr_rl)
    print("\nResults exported to simulation_results.csv")
