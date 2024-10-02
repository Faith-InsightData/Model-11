import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Step 1: Load Dataset
def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv('C:/Users/HP/OneDrive/Documents/Model 11/model_training/heart-disease.csv')
    return df

# Step 2: Prepare the dataset
def prepare_data(df):
    """
    Split the dataset into features and target, then into training and testing sets.
    """
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 3: Train the model
def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier on the training data.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data and print the results.
    """
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the model as a pickle file
def save_model(model, filename='heart_disease_model.pkl'):
    """
    Save the trained model to a file using pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as {filename}")

def main():
    """
    Main function to load data, prepare it, train the model, evaluate it, and save the model.
    """
    file_path = 'C:/Users/HP/OneDrive/Documents/Model 11/model_training/heart-disease.csv'
    df = load_data('C:/Users/HP/OneDrive/Documents/Model 11/model_training/heart-disease.csv')
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

if __name__ == '__main__':
    main()
