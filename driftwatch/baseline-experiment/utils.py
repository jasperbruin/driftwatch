import numpy as np

def apply_controlled_drift(data, drift_strength):
    """
    Apply a controlled amount of drift to the data.

    Args:
        data: The dataset to modify
        drift_strength: Float between 0 (no drift) and 10 (maximum drift)

    Returns:
        Modified dataset with controlled drift
    """
    if drift_strength == 0:
        return data.copy()  # No drift

    # Normalize drift_strength to a percentage of maximum shift
    drift_percentage = drift_strength / 10.0
    shifted_data = data.copy()

    # Apply different types of drift based on strength
    # For temporal features, we'll shift the values more significantly
    if 'year' in shifted_data.columns:
        year_nunique = shifted_data['year'].nunique()
        # More aggressive shift based on drift strength
        shift_amount = max(1, int(year_nunique * drift_percentage * 0.8))
        # Randomly shift some portion of the data based on drift strength
        mask = np.random.rand(len(shifted_data)) < drift_percentage
        shifted_data.loc[mask, 'year'] = (shifted_data.loc[mask, 'year'] + shift_amount) % year_nunique

    if 'month' in shifted_data.columns:
        month_nunique = shifted_data['month'].nunique()
        shift_amount = max(1, int(month_nunique * drift_percentage * 0.5))
        mask = np.random.rand(len(shifted_data)) < drift_percentage
        shifted_data.loc[mask, 'month'] = (shifted_data.loc[mask, 'month'] + shift_amount) % month_nunique

    if 'day' in shifted_data.columns:
        day_nunique = shifted_data['day'].nunique()
        shift_amount = max(1, int(day_nunique * drift_percentage * 0.3))
        mask = np.random.rand(len(shifted_data)) < drift_percentage
        shifted_data.loc[mask, 'day'] = (shifted_data.loc[mask, 'day'] + shift_amount) % day_nunique

    # Introduce drift in user_id distribution for higher drift strengths
    if 'user_id' in shifted_data.columns and drift_strength > 3:
        user_nunique = shifted_data['user_id'].nunique()
        unique_users = shifted_data['user_id'].unique()

        # Select a percentage of users to modify based on drift strength
        users_to_change = int(user_nunique * (drift_percentage - 0.3))
        if users_to_change > 0:
            selected_users = np.random.choice(unique_users, size=users_to_change, replace=False)
            new_values = (selected_users + np.random.randint(1, user_nunique // 2, size=users_to_change)) % user_nunique

            for old, new in zip(selected_users, new_values):
                shifted_data.loc[shifted_data['user_id'] == old, 'user_id'] = new

    # Debug print to verify drift is being applied
    print(f"Applied drift with strength {drift_strength}: "
          f"Modified {(shifted_data != data).any(axis=1).sum()} of {len(data)} records")

    return shifted_data