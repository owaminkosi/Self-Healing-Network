import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import time

# =====================================================
# 1️⃣ CREATE DUAL STAR TOPOLOGY
# =====================================================

def create_dual_star():
    G = nx.Graph()

    primary_hub = 1
    backup_hub = 2

    for i in range(1, 9):
        G.add_node(i, status="active", role="normal")

    G.nodes[primary_hub]["role"] = "primary"
    G.nodes[backup_hub]["role"] = "backup"

    # Connect all other nodes to both hubs
    for node in range(3, 9):
        G.add_edge(primary_hub, node, status="active", load=0)
        G.add_edge(backup_hub, node, status="active", load=0)

    # Connect primary and backup hub
    G.add_edge(primary_hub, backup_hub, status="active", load=0)

    return G


# =====================================================
# 2️⃣ VISUALIZATION
# =====================================================

def draw_network(G, title="Network"):
    pos = {}
    pos[1] = (0, 0)
    pos[2] = (0, -2)

    angle = 0
    radius = 4
    step = 360 / (len(G.nodes()) - 2)

    for node in G.nodes():
        if node not in [1, 2]:
            x = radius * math.cos(math.radians(angle))
            y = radius * math.sin(math.radians(angle))
            pos[node] = (x, y)
            angle += step

    node_colors = []
    for n in G.nodes():
        if G.nodes[n]["status"] == "failed":
            node_colors.append("red")
        elif G.nodes[n]["role"] == "primary":
            node_colors.append("blue")
        elif G.nodes[n]["role"] == "backup":
            node_colors.append("orange")
        else:
            node_colors.append("lightgreen")

    edge_colors = []
    for e in G.edges():
        if G.edges[e]["status"] == "failed":
            edge_colors.append("red")
        elif G.edges[e]["load"] > 7:
            edge_colors.append("purple")
        else:
            edge_colors.append("black")

    nx.draw(G, pos, with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=1000,
            width=2)

    plt.title(title)
    plt.show()


# =====================================================
# 3️⃣ FAILURE INJECTION
# =====================================================

def fail_node(G, node):
    G.nodes[node]["status"] = "failed"
    print(f"Node {node} FAILED")


# =====================================================
# 4️⃣ HUB FAILURE DETECTION & ELECTION
# =====================================================

def detect_and_heal_hub(G):

    primary = [n for n,d in G.nodes(data=True) if d["role"]=="primary"][0]

    if G.nodes[primary]["status"] == "failed":
        print("Primary hub failed!")

        # Promote backup
        backup_nodes = [n for n,d in G.nodes(data=True) if d["role"]=="backup"]
        if backup_nodes:
            new_primary = backup_nodes[0]
            G.nodes[new_primary]["role"] = "primary"
            print(f"Backup node {new_primary} promoted to PRIMARY")
        else:
            # Elect node with highest degree
            candidates = [n for n,d in G.nodes(data=True) if d["status"]=="active"]
            if candidates:
                best = max(candidates, key=lambda n: G.degree(n))
                G.nodes[best]["role"] = "primary"
                print(f"Node {best} elected as new PRIMARY")


# =====================================================
# 5️⃣ CONGESTION SIMULATION
# =====================================================

def simulate_congestion(G):
    for e in G.edges():
        if G.edges[e]["status"] == "active":
            G.edges[e]["load"] = random.randint(0,10)


# =====================================================
# 6️⃣ REINFORCEMENT LEARNING (Q-LEARNING)
# =====================================================

Q_table = {}

actions = ["reroute", "reduce_load", "do_nothing"]

def get_state(G):
    connectivity = nx.is_connected(G.subgraph(
        [n for n,d in G.nodes(data=True) if d["status"]=="active"]
    ))

    congestion = any(G.edges[e]["load"] > 7 for e in G.edges())

    return (connectivity, congestion)


def choose_action(state, epsilon=0.2):
    if random.uniform(0,1) < epsilon:
        return random.choice(actions)

    if state not in Q_table:
        Q_table[state] = {a:0 for a in actions}

    return max(Q_table[state], key=Q_table[state].get)


def update_q(state, action, reward, alpha=0.1, gamma=0.9):
    if state not in Q_table:
        Q_table[state] = {a:0 for a in actions}

    current_q = Q_table[state][action]
    Q_table[state][action] = current_q + alpha * (reward + gamma * max(Q_table[state].values()) - current_q)


def apply_action(G, action):
    reward = 0

    if action == "reroute":
        reward += 5

    elif action == "reduce_load":
        for e in G.edges():
            G.edges[e]["load"] = max(0, G.edges[e]["load"] - 5)
        reward += 10

    elif action == "do_nothing":
        reward -= 5

    return reward


# =====================================================
# 7️⃣ FULL SELF-HEALING SIMULATION LOOP
# =====================================================

def simulate():

    G = create_dual_star()
    draw_network(G, "Initial Dual-Star Network")

    for episode in range(5):

        print("\n--- Episode", episode+1, "---")

        simulate_congestion(G)

        # Randomly fail primary sometimes
        if random.random() < 0.4:
            fail_node(G, 1)

        detect_and_heal_hub(G)

        state = get_state(G)
        action = choose_action(state)

        reward = apply_action(G, action)

        update_q(state, action, reward)

        print("State:", state)
        print("Action:", action)
        print("Reward:", reward)

        draw_network(G, f"Episode {episode+1}")

        time.sleep(1)


if __name__ == "__main__":
    simulate()
