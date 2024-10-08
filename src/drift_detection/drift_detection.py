from river import drift

def detect_drift(error_stream):
    adwin = drift.ADWIN(delta=0.002)
    for i, error in enumerate(error_stream):
        adwin.update(error)
        if adwin.change_detected:
            print(f"Change detected at index {i}")
            # Trigger retraining
