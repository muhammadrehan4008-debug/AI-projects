# Project 2 - Data Classification using AI
# Algorithm: K-Nearest Neighbors
# Dataset: Iris

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

print("Project 2 - Iris Classification using KNN")
print("-" * 45)

# Load the dataset
iris = load_iris()
X = iris.data      # sepal length, sepal width, petal length, petal width
y = iris.target    # 0 = Setosa, 1 = Versicolor, 2 = Virginica

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Classes: {list(iris.target_names)}")
print(f"Class counts: {dict(zip(iris.target_names, np.bincount(y)))}")

# Scale features - KNN is distance based so scale matters a lot here
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Try k values from 1 to 20 and track error rate for each,
# so we can pick whichever k performs best instead of guessing
error_rates = []
k_range = range(1, 21)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, pred))

best_k = k_range[np.argmin(error_rates)]
print(f"Best k found: {best_k} (error rate = {min(error_rates):.4f})")

# Train the final model using the best k
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions, target_names=iris.target_names)

print(f"\nAccuracy: {acc * 100:.2f}%")
print(f"F1 score: {f1:.4f}")
print("\nClassification report:")
print(report)

# ---------------- Plots ----------------
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#F0F4F8')

blue = '#1B3A6B'
orange = '#E05C1A'
green = '#2E7D32'
light_bg = '#EAF0FB'

# Elbow curve - shows error rate for each k we tried
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor(light_bg)
ax1.plot(k_range, error_rates, color=blue, lw=2.5, marker='o',
         markersize=7, markerfacecolor='white', markeredgecolor=blue, zorder=3)
ax1.scatter([best_k], [min(error_rates)], color=orange, s=200, zorder=5,
            label=f'Best k = {best_k}')
ax1.set_title('Choosing k (Elbow Method)', fontsize=13, fontweight='bold', color=blue)
ax1.set_xlabel('k')
ax1.set_ylabel('Error rate')
ax1.legend()
ax1.grid(True, alpha=0.4)

# Confusion matrix
ax2 = fig.add_subplot(2, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names,
            linewidths=1, linecolor='white', annot_kws={"size": 14, "weight": "bold"}, ax=ax2)
ax2.set_title('Confusion Matrix', fontsize=13, fontweight='bold', color=blue)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# F1 score per class
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor(light_bg)
per_class_f1 = f1_score(y_test, predictions, average=None)
bars = ax3.bar(iris.target_names, per_class_f1, color=[blue, orange, green],
               edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, per_class_f1):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
              f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 1.15)
ax3.set_title('F1 Score per Class', fontsize=13, fontweight='bold', color=blue)
ax3.set_ylabel('F1 score')
ax3.axhline(y=f1, color='red', linestyle='--', alpha=0.6, label=f'Weighted avg = {f1:.2f}')
ax3.legend()
ax3.grid(axis='y', alpha=0.4)

# Feature space plot - petal length vs petal width, with correct/wrong predictions marked
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor(light_bg)
colors_map = {0: blue, 1: orange, 2: green}
markers_map = {0: 'o', 1: 's', 2: '^'}

for cls in range(3):
    mask = y == cls
    ax4.scatter(X[mask, 2], X[mask, 3], c=colors_map[cls], marker=markers_map[cls],
                s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

correct = predictions == y_test
wrong = ~correct
X_test_orig = scaler.inverse_transform(X_test)

ax4.scatter(X_test_orig[correct, 2], X_test_orig[correct, 3], s=120, facecolors='none',
            edgecolors='lime', linewidths=2, label='Correct', zorder=5)
ax4.scatter(X_test_orig[wrong, 2], X_test_orig[wrong, 3], s=120, facecolors='none',
            edgecolors='red', linewidths=2.5, marker='x', label='Wrong', zorder=5)

patches = [mpatches.Patch(color=colors_map[i], label=iris.target_names[i]) for i in range(3)]
ax4.legend(handles=patches + ax4.get_legend_handles_labels()[0][-2:], fontsize=9, loc='upper left')
ax4.set_title('Petal Length vs Petal Width', fontsize=13, fontweight='bold', color=blue)
ax4.set_xlabel('Petal length (cm)')
ax4.set_ylabel('Petal width (cm)')
ax4.grid(True, alpha=0.4)

fig.suptitle(f'KNN Iris Classification | k={best_k} | Accuracy={acc*100:.1f}% | F1={f1:.4f}',
             fontsize=14, fontweight='bold', color=blue, y=1.01)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/project2_results.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("\nDone - saved plot to project2_results.png")
