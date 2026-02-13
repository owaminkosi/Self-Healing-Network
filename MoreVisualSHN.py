import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import csv
import os

# -------------------------
# Global metrics
# -------------------------
metrics_log = []

# -------------------------
# Dual-Star Topology
# -------------------------
def create_dual_star(num_nodes=10):
    G = nx.Graph()
    for i in range(1, num_nodes+1):
        G.add_node(i, status="active")
    primary, backup = 1, 2
    for node in range(3, num_nodes+1):
        G.add_edge(primary, node, load=0)
        G.add_edge(backup, node, load=0)
    G.add_edge(primary, backup, load=0)
    return G

# -------------------------
# Visualize Network
# -------------------------
def draw_network(G, title="Network"):
    pos = nx.spring_layout(G, seed=42)
    node_colors = []
    for n in G.nodes():
        role = G.nodes[n].get("role","normal")
        status = G.nodes[n]["status"]
        if status == "failed":
            node_colors.append("red")
        elif role == "primary":
            node_colors.append("blue")
        elif role == "backup":
            node_colors.append("orange")
        else:
            node_colors.append("lightgreen")
    edge_colors = []
    widths = []
    for u,v in G.edges():
        load = G.edges[(u,v)].get("load",0)
        widths.append(1 + load*0.3)
        if G.edges[(u,v)].get("status","active") == "failed":
            edge_colors.append("red")
        elif load > 7:
            edge_colors.append("purple")
        else:
            edge_colors.append("black")
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            edge_color=edge_colors, width=widths, node_size=800)
    plt.title(title)
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()

# -------------------------
# Packet Flow Simulation
# -------------------------
def simulate_packets(G, num_packets=5):
    sent = num_packets
    delivered = 0
    total_delay = 0
    active_nodes = [n for n,d in G.nodes(data=True) if d["status"]=="active"]
    if len(active_nodes)<2:
        return sent, delivered, sent-delivered, 0
    for _ in range(num_packets):
        src,dst = random.sample(active_nodes,2)
        try:
            path = nx.shortest_path(G, src, dst)
            delay = sum(G.edges[(path[i], path[i+1])]["load"]+1 for i in range(len(path)-1))
            delivered += 1
            total_delay += delay
        except:
            pass
    loss = sent - delivered
    avg_delay = total_delay/delivered if delivered else 0
    return sent, delivered, loss, avg_delay

# -------------------------
# Congestion & Failures
# -------------------------
def random_congestion(G):
    for e in G.edges():
        G.edges[e]["load"] = max(0, G.edges[e].get("load",0)+random.randint(0,3))

def fail_node(G, node):
    G.nodes[node]["status"]="failed"
    print(f"Node {node} FAILED")

def cascading_failure(G, threshold=12):
    for e in list(G.edges()):
        if G.edges[e]["load"]>threshold:
            G.remove_edge(*e)
            for node in e:
                if random.random()<0.3:
                    G.nodes[node]["status"]="failed"

# -------------------------
# SDN Controller
# -------------------------
class SDNController:
    def __init__(self, mode="RL"):
        self.mode = mode

    def monitor(self,G):
        connected = nx.is_connected(G.subgraph([n for n,d in G.nodes(data=True) if d["status"]=="active"]))
        congestion = any(G.edges[e].get("load",0)>7 for e in G.edges())
        return (connected, congestion)

    def decide(self,state):
        if self.mode=="traditional":
            return "reroute"
        return random.choice(["reroute","reduce_load","do_nothing"])

    def enforce(self,G,action):
        reward=0
        if action=="reroute":
            reward+=5
        elif action=="reduce_load":
            for e in G.edges():
                G.edges[e]["load"]=max(0,G.edges[e]["load"]-5)
            reward+=10
        elif action=="do_nothing":
            reward-=5
        return reward

# -------------------------
# Simulation Loop
# -------------------------
def run_simulation(num_nodes=10,steps=15,mode="RL"):
    G=create_dual_star(num_nodes)
    controller=SDNController(mode)
    for t in range(steps):
        random_congestion(G)
        if t==3: fail_node(G,1)          # Primary hub fails
        if t==6: cascading_failure(G)    # Cascading failure
        state=controller.monitor(G)
        action=controller.decide(state)
        reward=controller.enforce(G,action)
        sent,delivered,loss,delay=simulate_packets(G)
        metrics_log.append([mode,t,sent,delivered,loss,delay,state[0]])
        draw_network(G,f"{mode} - Step {t}")
    # Save CSV
    filename=os.path.join(os.getcwd(),f"simulation_results_{mode}.csv")
    with open(filename,"w",newline="") as f:
        import csv
        writer=csv.writer(f)
        writer.writerow(["Mode","Time","Sent","Delivered","Loss","AvgDelay","Connected"])
        writer.writerows(metrics_log)
    print(f"Simulation finished! CSV saved: {filename}")

# -------------------------
# Run Simulation
# -------------------------
if __name__=="__main__":
    run_simulation(mode="RL")
    run_simulation(mode="traditional")
