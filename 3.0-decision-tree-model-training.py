import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import pickle
import matplotlib.pyplot as plt

# Step 1: Load Dataset
def load_data():
    # Load your dataset here (ensure the file path is correct)
    df = pd.read_csv(r'C:\Users\HP\OneDrive\Documents\Model 11\heart-disease.csv')
    return df

# Step 2: Prepare the dataset
def prepare_data(df):
    # Split data into features (X) and target (y)
    X = df.drop(columns=['target'])
    y = df['target']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Step 3: Train the model
def train_model(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Print classification report and accuracy
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 5: Save the model as a pickle file
def save_model(model, filename='decision_tree_heart_disease_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as {filename}")

# Step 6: Save the decision tree as an image
def save_tree_image(model, feature_names, filename='decision_tree_plot.png'):
    plt.figure(figsize=(20, 10))  # Adjust the size for better readability
    plot_tree(model, feature_names=feature_names, class_names=['No Disease', 'Disease'], filled=True, rounded=True)
    plt.savefig(filename)
    print(f"Decision Tree image saved as {filename}")

def main():
    try:
        # Load data
        df = load_data()

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(df)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        evaluate_model(model, X_test, y_test)

        # Save model
        save_model(model)

        # Save decision tree as image
        save_tree_image(model, feature_names=X_train.columns)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting gracefully.")
    finally:
        print("Cleanup if necessary.")

if __name__ == "__main__":
    main()
