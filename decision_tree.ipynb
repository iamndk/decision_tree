# Decision tree for classification

# Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualize Decision Tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()



# Decision_treee For Regression

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
import plotly.graph_objects as go

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train tree
reg = DecisionTreeRegressor(max_depth=3, random_state=42)
reg.fit(X_train, y_train)

# Get tree rules
tree_rules = export_text(reg, feature_names=feature_names)
print(tree_rules)

# Create a simple interactive tree visualization using Plotly
def plot_tree_interactive(tree, feature_names):
    from sklearn import tree as sktree
    import networkx as nx

    G = nx.DiGraph()
    node_info = {}

    def recurse(node_id, parent=None, edge_label=""):
        if node_id == -1:
            return
        node = tree.tree_
        name = f"node{node_id}"
        if parent is not None:
            G.add_edge(parent, name, label=edge_label)
        if node.children_left[node_id] == node.children_right[node_id]:
            label = f"Leaf: {node.value[node_id][0][0]:.2f}"
        else:
            feat = feature_names[node.feature[node_id]]
            thresh = node.threshold[node_id]
            label = f"{feat} <= {thresh:.2f}<br>Samples: {node.n_node_samples}"
            recurse(node.children_left[node_id], name, "True")
            recurse(node.children_right[node_id], name, "False")
        node_info[name] = label

    recurse(0)
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=2, color='black'), hoverinfo='none')

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y_ = pos[node]
        node_x.append(x)
        node_y.append(y_)
        node_text.append(node_info[node])

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
        marker=dict(size=50, color='lightblue'), hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False)
    fig.show()

plot_tree_interactive(reg, feature_names)
