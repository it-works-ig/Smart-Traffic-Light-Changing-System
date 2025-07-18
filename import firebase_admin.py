import firebase_admin
from firebase_admin import credentials, firestore
import time

# Load Firebase credentials
cred = credentials.Certificate(r"C:\Users\Jayesh\Desktop\yolo\yolo-cf82f-firebase-adminsdk-fbsvc-92d43984b9.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

def get_traffic_status():
    """Fetches the latest traffic status from Firebase Firestore."""
    doc_ref = db.collection("traffic_status").document("latest")
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("traffic_status", "Unknown")
    return "Unknown"
3
def change_signal_color(status):
    """Changes traffic signal color based on traffic status."""
    if status == "Traffic Jam":
        print("🚦 Changing signal to GREEN to clear traffic")

    else:
        print("🚦 No change in signal color")

if __name__ == "__main__":
    while True:
        traffic_status = get_traffic_status()
        print(f"Current Traffic Status: {traffic_status}")
        change_signal_color(traffic_status)
        time.sleep(10)  # Check every 10 seconds
