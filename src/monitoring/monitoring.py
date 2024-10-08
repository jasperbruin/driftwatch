from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

def generate_data_drift_report(reference_data, current_data):
    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(reference_data, current_data)
    dashboard.save("reports/data_drift_report.html")
