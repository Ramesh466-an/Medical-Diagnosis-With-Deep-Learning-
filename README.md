# =========================================================================
# Project: Medical Diagnosis with Deep Learning and DSA Tools
#
# This script simulates a medical diagnosis system using core data structures
# and a conceptual deep learning model.
#
# Key Concepts:
#   1. Hash Map (Dictionary): For efficient storage and retrieval of patient data.
#   2. Queue: To process patient diagnoses in a First-In, First-Out (FIFO) order.
#   3. Stack: To maintain a history of completed diagnoses for review or "undo" functionality.
#   4. Deep Learning Model (Conceptual): A simplified function that simulates a model
#      making a binary classification (e.g., diagnosing a disease).
# =========================================================================

# --- 1. Data Structures and Initial Data Setup ---

# A Hash Map (Python dictionary) for patient records.
# Key: patient_id, Value: patient_data dictionary.
patient_database = {}

# A Queue (Python list) for processing new diagnosis requests.
# We use a list and treat it like a queue with append() and pop(0).
diagnosis_queue = []

# A Stack (Python list) to store the history of diagnoses.
# We use a list and treat it like a stack with append() and pop().
diagnosis_history = []

# =========================================================================
# SIMULATED DATA
# Each record represents a patient with a unique ID and medical features.
# A 'label' is included for training purposes in a real deep learning project.
# In this simulation, we'll use it to check our 'model's' performance.
# =========================================================================
initial_patients = [
    {'id': 'P001', 'features': [0.8, 0.2, 0.9, 0.5], 'label': 1},  # High risk
    {'id': 'P002', 'features': [0.1, 0.9, 0.2, 0.4], 'label': 0},  # Low risk
    {'id': 'P003', 'features': [0.7, 0.3, 0.8, 0.6], 'label': 1},  # High risk
    {'id': 'P004', 'features': [0.2, 0.8, 0.3, 0.1], 'label': 0},  # Low risk
    {'id': 'P005', 'features': [0.9, 0.1, 0.7, 0.8], 'label': 1},  # High risk
]

def add_patients_to_database(patients):
    """
    Adds a list of patient records to the patient database (hash map).
    """
    for patient in patients:
        patient_database[patient['id']] = patient
        print(f"Added patient {patient['id']} to the database.")

def enqueue_for_diagnosis(patient_id):
    """
    Adds a patient to the diagnosis queue.
    """
    if patient_id in patient_database:
        diagnosis_queue.append(patient_id)
        print(f"Enqueued patient {patient_id} for diagnosis.")
    else:
        print(f"Error: Patient {patient_id} not found in database.")

# --- 2. Conceptual Deep Learning Model ---

def conceptual_deep_learning_model(features):
    """
    A simplified function that simulates a deep learning model.
    It performs a simple weighted sum to determine a diagnosis score.
    In a real project, this would be a complex model (e.g., TensorFlow, PyTorch).
    """
    # A simple conceptual "weighted sum" model.
    # The weights are pre-determined for this simulation.
    weights = [0.5, -0.4, 0.6, 0.3]
    bias = -0.2

    score = sum(f * w for f, w in zip(features, weights)) + bias
    
    # Apply a sigmoid-like activation function for a binary result.
    prediction = 1 if score > 0 else 0
    return prediction, score

# --- 3. Diagnosis and DSA Operations ---

def run_diagnosis():
    """
    Processes patients from the queue and adds the diagnosis to the history stack.
    """
    if not diagnosis_queue:
        print("No patients in the queue to diagnose.")
        return

    # Dequeue the first patient
    patient_id = diagnosis_queue.pop(0)
    patient = patient_database[patient_id]

    print(f"Processing diagnosis for patient {patient_id}...")
    
    # Run the conceptual deep learning model on the patient's features
    prediction, score = conceptual_deep_learning_model(patient['features'])

    result = {
        'patient_id': patient_id,
        'prediction': prediction,
        'confidence_score': score,
        'timestamp': 'current_time_placeholder' # In a real app, use datetime
    }
    
    # Push the result to the diagnosis history stack
    diagnosis_history.append(result)
    
    print(f"Diagnosis complete for {patient_id}. Result pushed to history.")
    print(f"Predicted Diagnosis: {'High Risk' if prediction == 1 else 'Low Risk'} (Score: {score:.2f})")
    
def review_last_diagnosis():
    """
    Pops the last diagnosis from the stack for review.
    """
    if not diagnosis_history:
        print("No diagnoses in history to review.")
        return

    # Pop the last diagnosis from the stack
    last_diagnosis = diagnosis_history.pop()
    print("\n--- Reviewing Last Diagnosis ---")
    print(f"Patient ID: {last_diagnosis['patient_id']}")
    print(f"Prediction: {'High Risk' if last_diagnosis['prediction'] == 1 else 'Low Risk'}")
    print(f"Confidence Score: {last_diagnosis['confidence_score']:.2f}")
    print("--------------------------------")

# --- 4. Main Execution ---

if __name__ == "__main__":
    print("--- Medical Diagnosis System Startup ---")

    # Step 1: Add patients to the Hash Map
    add_patients_to_database(initial_patients)

    print("\n--- Enqueuing Patients for Diagnosis ---")
    # Step 2: Enqueue some patients for diagnosis
    enqueue_for_diagnosis('P001')
    enqueue_for_diagnosis('P003')
    enqueue_for_diagnosis('P002')

    print("\n--- Running Diagnosis Process ---")
    # Step 3: Run the diagnosis process repeatedly to clear the queue
    run_diagnosis()
    run_diagnosis()
    run_diagnosis()

    # Step 4: Check the diagnosis history (Stack)
    print(f"\nDiagnosis History has {len(diagnosis_history)} entries.")

    # Step 5: Review the last diagnosis from the stack
    review_last_diagnosis()
    
    print(f"\nDiagnosis History now has {len(diagnosis_history)} entries.")

    # Step 6: Attempt to review again to see the stack behavior
    review_last_diagnosis()

