import sys

# Open a file to save the output
with open("wandb_results.txt", "w", encoding="utf-8") as f:
    # Redirect standard output to the file
    sys.stdout = f
    
    import wandb
    import pandas as pd
    import yaml
    api = wandb.Api()
    project_path = "project_folder"
    project_runs = api.runs(project_path, per_page=1000)


    def update_yaml_config(name, new_config, file_path="final_config.yaml"):
        # Load existing YAML file
        try:
            with open(file_path, "r") as file:
                config = yaml.safe_load(file) or {}
        except FileNotFoundError:
            config = {}

        # Define the parameter name
        param_name = f"{name}_parameter"

        # Update the existing parameter or create a new one
        config[param_name] = {
            "alpha": new_config.get("alpha"),
            "epochs": new_config.get("epochs"),
            "batch_size": new_config.get("batch_size"),
            "temperature": new_config.get("temperature"),
            "learning_rate": new_config.get("learning_rate"),
            "warmup_epochs": new_config.get("warmup_epochs"),
            "schedule_type": {
                "distil": new_config.get("schedule_type_distil"),
                "student": new_config.get("schedule_type_student"),
                "teacher": new_config.get("schedule_type_teacher"),
            },
        }

        # Write updated config back to YAML file
        with open(file_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)

    # Convert to DataFrame for easier filtering
    runs_df = pd.DataFrame()
    for run in project_runs:
        # Get the summary metrics
        summary = run.summary._json_dict
        
        # Create a record with run info and metrics
        run_data = {
            'run_id': run.id,
            'run_name': run.name,
            'teacher_val_accuracy': summary.get('teacher_val_accuracy', None)  # Store as val_rmse for clarity
        }
        runs_df = pd.concat([runs_df, pd.DataFrame([run_data])], ignore_index=True)

    # Convert teacher_val_accuracy to numeric to ensure proper filtering
    runs_df['teacher_val_accuracy'] = pd.to_numeric(runs_df['teacher_val_accuracy'], errors='coerce')

    # Filter runs with teacher_val_accuracy above 83.0
    filtered_runs = runs_df[runs_df['teacher_val_accuracy'] > 83.0]

    # Sort by teacher_val_accuracy in descending order to see highest values first
    filtered_runs = filtered_runs.sort_values(by='teacher_val_accuracy', ascending=False)

    # Display the filtered runs
    print("Runs with teacher_val_accuracy > 83.0:")
    print(filtered_runs)

    # If you need the actual run object for a specific run
    if len(filtered_runs) > 0:
        specific_run_id = filtered_runs.iloc[0]['run_id']  # Get run with highest val_rmse
        print(f"\nFetching details for run with highest teacher_val_accuracy: {specific_run_id}")
        run = api.run(f"{project_path}/{specific_run_id}")
        # Print some details about the run
        print(f"Run name: {run.name}")
        print(f"Val accuracy: {run.summary.get('teacher_val_accuracy')}")
        
        # If you want to access more details or files from this run
        print("Config:", run.config)
        # files = run.files()
        # print("Files in this run:", [f.name for f in files])
    else:
        print("\nNo runs found with teacher_val_accuracy > 83.0")


    print("\n************************************\n")
    print("\nStudent")

    # Convert to DataFrame for easier filtering
    runs_df = pd.DataFrame()
    for run in project_runs:
        # Get the summary metrics
        summary = run.summary._json_dict
        
        # Create a record with run info and metrics
        run_data = {
            'run_id': run.id,
            'run_name': run.name,
            'student_val_accuracy': summary.get('student_val_accuracy', None)  # Store as val_rmse for clarity
        }
        runs_df = pd.concat([runs_df, pd.DataFrame([run_data])], ignore_index=True)

    # Convert teacher_val_accuracy to numeric to ensure proper filtering
    runs_df['student_val_accuracy'] = pd.to_numeric(runs_df['student_val_accuracy'], errors='coerce')

    # Filter runs with teacher_val_accuracy above 83.0
    filtered_runs = runs_df[runs_df['student_val_accuracy'] > 80.5]

    # Sort by teacher_val_accuracy in descending order to see highest values first
    filtered_runs = filtered_runs.sort_values(by='student_val_accuracy', ascending=False)

    # Display the filtered runs
    print("Runs with student_val_accuracy > 80.5:")
    print(filtered_runs)

    if len(filtered_runs) > 0:
        specific_run_id = filtered_runs.iloc[0]['run_id']  # Get run with highest val_rmse
        print(f"\nFetching details for run with highest student_val_accuracy: {specific_run_id}")
        run = api.run(f"{project_path}/{specific_run_id}")
        # Print some details about the run
        print(f"Run name: {run.name}")
        print(f"Val accuracy: {run.summary.get('student_val_accuracy')}")
        
        # If you want to access more details or files from this run
        print("Config:", run.config)
        # files = run.files()
        # print("Files in this run:", [f.name for f in files])
    else:
        print("\nNo runs found with teacher_val_accuracy > 80.5")


    print("\n************************************\n")
    print("\nDistil")

    # Convert to DataFrame for easier filtering
    runs_df = pd.DataFrame()
    for run in project_runs:
        # Get the summary metrics
        summary = run.summary._json_dict
        
        # Create a record with run info and metrics
        run_data = {
            'run_id': run.id,
            'run_name': run.name,
            'distill_val_accuracy': summary.get('distill_val_accuracy', None)  # Store as val_rmse for clarity
        }
        runs_df = pd.concat([runs_df, pd.DataFrame([run_data])], ignore_index=True)

    # Convert teacher_val_accuracy to numeric to ensure proper filtering
    runs_df['distill_val_accuracy'] = pd.to_numeric(runs_df['distill_val_accuracy'], errors='coerce')

    # Filter runs with teacher_val_accuracy above 82.0
    filtered_runs = runs_df[runs_df['distill_val_accuracy'] > 82.0]

    # Sort by teacher_val_accuracy in descending order to see highest values first
    filtered_runs = filtered_runs.sort_values(by='distill_val_accuracy', ascending=False)

    # Display the filtered runs
    print("Runs with distill_val_accuracy > 82.0:")
    print(filtered_runs)

    # If you need the actual run object for a specific run
    if len(filtered_runs) > 0:
        specific_run_id = filtered_runs.iloc[0]['run_id']  # Get run with highest val_rmse
        print(f"\nFetching details for run with highest distill_val_accuracy: {specific_run_id}")
        run = api.run(f"{project_path}/{specific_run_id}")
        # Print some details about the run
        print(f"Run name: {run.name}")
        print(f"Val accuracy: {run.summary.get('distill_val_accuracy')}")
        
        # If you want to access more details or files from this run
        print("Config:", run.config)
        # files = run.files()
        # print("Files in this run:", [f.name for f in files])
    else:
        print("\nNo runs found with distill_val_accuracy > 82.0")
        
        
        
        
        
        
        
        
        
        
        
        
        
    print("\n************")
    print("based on the max")
    print("\n************")

    print("\n*********************************")
    print("\nTeacher")

    name = "teacher"

    # Create DataFrame to store results
    results_df = pd.DataFrame(columns=[
        'run_id', 'run_name', 
        f'final_{name}_val_accuracy', f'max_{name}_val_accuracy', 
        'max_accuracy_step', 'total_steps'
    ])

    # Process each run
    for run in project_runs:
        try:
            # Get run history for teacher_val_accuracy
            history = run.scan_history(keys=[f'{name}_val_accuracy'])
            
            # Convert to DataFrame for easier analysis
            history_df = pd.DataFrame(list(history))
            
            # Skip if no teacher_val_accuracy data
            if f'{name}_val_accuracy' not in history_df.columns or history_df.empty:
                continue
                
            # Get final and maximum teacher_val_accuracy values
            globals()[f"final_{name}_val_accuracy"] = history_df[f'{name}_val_accuracy'].iloc[-1] if not history_df.empty else None
            globals()[f"max_{name}_val_accuracy"] = history_df[f'{name}_val_accuracy'].max() if not history_df.empty else None
            
            # Get the step where max teacher_val_accuracy occurred
            max_accuracy_step = None
            if not history_df.empty and f'{name}_val_accuracy' in history_df.columns:
                max_index = history_df[f'{name}_val_accuracy'].idxmax()
                max_accuracy_step = max_index + 1  # Add 1 since index is 0-based
                
            # Add to results
            results_df = pd.concat([results_df, pd.DataFrame([{
                'run_id': run.id,
                'run_name': run.name,
                f'final_{name}_val_accuracy': eval(f"final_{name}_val_accuracy"),
                f'max_{name}_val_accuracy': eval(f"max_{name}_val_accuracy"),
                'max_accuracy_step': max_accuracy_step,
                'total_steps': len(history_df)
            }])], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing run {run.id}: {e}")

    # Convert to numeric for proper sorting and filtering
    results_df[f'final_{name}_val_accuracy'] = pd.to_numeric(results_df[f'final_{name}_val_accuracy'], errors='coerce')
    results_df[f'max_{name}_val_accuracy'] = pd.to_numeric(results_df[f'max_{name}_val_accuracy'], errors='coerce')

    # Filter runs with max teacher_val_accuracy above 83.0
    filtered_results = results_df[results_df[f'max_{name}_val_accuracy'] > 85.0]

    # Sort by max_teacher_val_accuracy in descending order
    filtered_results = filtered_results.sort_values(by=f'max_{name}_val_accuracy', ascending=False)

    # Display runs that achieved high teacher_val_accuracy during training
    print(f"Runs that achieved {name}_val_accuracy > 83.0 during training:")
    # print(filtered_results)

    # Find runs where max_teacher_val_accuracy is significantly higher than final_teacher_val_accuracy
    # This indicates runs that peaked high but ended lower
    results_df['accuracy_drop'] = results_df[f'max_{name}_val_accuracy'] - results_df[f'final_{name}_val_accuracy']
    peaked_runs = results_df[results_df['accuracy_drop'] > 0.5].sort_values(by='accuracy_drop', ascending=False)

    print(f"\nRuns with significant drop in {name}_val_accuracy (peaked high but ended lower):")
    print(peaked_runs[['run_id', 'run_name', f'max_{name}_val_accuracy', f'final_{name}_val_accuracy', 'accuracy_drop', 'max_accuracy_step', 'total_steps']])

    # If you want details on the run with the highest max teacher_val_accuracy
    if len(filtered_results) > 0:
        top_run_id = filtered_results.iloc[0]['run_id']
        print(f"\nFetching details for run with highest max {name}_val_accuracy: {top_run_id}")
        run = api.run(f"{project_path}/{top_run_id}")
        print(f"Run name: {run.name}")
        print(f"Max {name}_val_accuracy: {filtered_results.iloc[0][f'max_{name}_val_accuracy']}")
        print(f"Final {name}_val_accuracy: {filtered_results.iloc[0][f'final_{name}_val_accuracy']}")
        print(f"Occurred at step: {filtered_results.iloc[0]['max_accuracy_step']} of {filtered_results.iloc[0]['total_steps']}")
        print("Config:", run.config)
        update_yaml_config(name, run.config)  # Updates "distill_parameter" in config.yaml




    print("\n*********************************")
    name = "student"

    print(f"\n{name.capitalize()}")



    # Create DataFrame to store results
    results_df = pd.DataFrame(columns=[
        'run_id', 'run_name', 
        f'final_{name}_val_accuracy', f'max_{name}_val_accuracy', 
        'max_accuracy_step', 'total_steps'
    ])

    # Process each run
    for run in project_runs:
        try:
            # Get run history for teacher_val_accuracy
            history = run.scan_history(keys=[f'{name}_val_accuracy'])
            
            # Convert to DataFrame for easier analysis
            history_df = pd.DataFrame(list(history))
            
            # Skip if no teacher_val_accuracy data
            if f'{name}_val_accuracy' not in history_df.columns or history_df.empty:
                continue
                
            # Get final and maximum teacher_val_accuracy values
            globals()[f"final_{name}_val_accuracy"] = history_df[f'{name}_val_accuracy'].iloc[-1] if not history_df.empty else None
            globals()[f"max_{name}_val_accuracy"] = history_df[f'{name}_val_accuracy'].max() if not history_df.empty else None
            
            # Get the step where max teacher_val_accuracy occurred
            max_accuracy_step = None
            if not history_df.empty and f'{name}_val_accuracy' in history_df.columns:
                max_index = history_df[f'{name}_val_accuracy'].idxmax()
                max_accuracy_step = max_index + 1  # Add 1 since index is 0-based
                
            # Add to results
            results_df = pd.concat([results_df, pd.DataFrame([{
                'run_id': run.id,
                'run_name': run.name,
                f'final_{name}_val_accuracy': eval(f"final_{name}_val_accuracy"),
                f'max_{name}_val_accuracy': eval(f"max_{name}_val_accuracy"),
                'max_accuracy_step': max_accuracy_step,
                'total_steps': len(history_df)
            }])], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing run {run.id}: {e}")

    # Convert to numeric for proper sorting and filtering
    results_df[f'final_{name}_val_accuracy'] = pd.to_numeric(results_df[f'final_{name}_val_accuracy'], errors='coerce')
    results_df[f'max_{name}_val_accuracy'] = pd.to_numeric(results_df[f'max_{name}_val_accuracy'], errors='coerce')

    # Filter runs with max teacher_val_accuracy above 80.0
    filtered_results = results_df[results_df[f'max_{name}_val_accuracy'] > 80.0]

    # Sort by max_teacher_val_accuracy in descending order
    filtered_results = filtered_results.sort_values(by=f'max_{name}_val_accuracy', ascending=False)

    # Display runs that achieved high teacher_val_accuracy during training
    print(f"Runs that achieved {name}_val_accuracy > 83.0 during training:")
    # print(filtered_results)

    # Find runs where max_teacher_val_accuracy is significantly higher than final_teacher_val_accuracy
    # This indicates runs that peaked high but ended lower
    results_df['accuracy_drop'] = results_df[f'max_{name}_val_accuracy'] - results_df[f'final_{name}_val_accuracy']
    peaked_runs = results_df[results_df['accuracy_drop'] > 0.5].sort_values(by='accuracy_drop', ascending=False)

    print(f"\nRuns with significant drop in {name}_val_accuracy (peaked high but ended lower):")
    print(peaked_runs[['run_id', 'run_name', f'max_{name}_val_accuracy', f'final_{name}_val_accuracy', 'accuracy_drop', 'max_accuracy_step', 'total_steps']])

    # If you want details on the run with the highest max teacher_val_accuracy
    if len(filtered_results) > 0:
        top_run_id = filtered_results.iloc[0]['run_id']
        print(f"\nFetching details for run with highest max {name}_val_accuracy: {top_run_id}")
        run = api.run(f"{project_path}/{top_run_id}")
        print(f"Run name: {run.name}")
        print(f"Max {name}_val_accuracy: {filtered_results.iloc[0][f'max_{name}_val_accuracy']}")
        print(f"Final {name}_val_accuracy: {filtered_results.iloc[0][f'final_{name}_val_accuracy']}")
        print(f"Occurred at step: {filtered_results.iloc[0]['max_accuracy_step']} of {filtered_results.iloc[0]['total_steps']}")
        print("Config:", run.config)
        update_yaml_config(name, run.config)  # Updates "distill_parameter" in config.yaml


    print("\n*********************************")
    name = "distill"

    print(f"\n{name.capitalize()}")



    # Create DataFrame to store results
    results_df = pd.DataFrame(columns=[
        'run_id', 'run_name', 
        f'final_{name}_val_accuracy', f'max_{name}_val_accuracy', 
        'max_accuracy_step', 'total_steps'
    ])

    # Process each run
    for run in project_runs:
        try:
            # Get run history for teacher_val_accuracy
            history = run.scan_history(keys=[f'{name}_val_accuracy'])
            
            # Convert to DataFrame for easier analysis
            history_df = pd.DataFrame(list(history))
            
            # Skip if no teacher_val_accuracy data
            if f'{name}_val_accuracy' not in history_df.columns or history_df.empty:
                continue
                
            # Get final and maximum teacher_val_accuracy values
            globals()[f"final_{name}_val_accuracy"] = history_df[f'{name}_val_accuracy'].iloc[-1] if not history_df.empty else None
            globals()[f"max_{name}_val_accuracy"] = history_df[f'{name}_val_accuracy'].max() if not history_df.empty else None
            
            # Get the step where max teacher_val_accuracy occurred
            max_accuracy_step = None
            if not history_df.empty and f'{name}_val_accuracy' in history_df.columns:
                max_index = history_df[f'{name}_val_accuracy'].idxmax()
                max_accuracy_step = max_index + 1  # Add 1 since index is 0-based
                
            # Add to results
            results_df = pd.concat([results_df, pd.DataFrame([{
                'run_id': run.id,
                'run_name': run.name,
                f'final_{name}_val_accuracy': eval(f"final_{name}_val_accuracy"),
                f'max_{name}_val_accuracy': eval(f"max_{name}_val_accuracy"),
                'max_accuracy_step': max_accuracy_step,
                'total_steps': len(history_df)
            }])], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing run {run.id}: {e}")

    # Convert to numeric for proper sorting and filtering
    results_df[f'final_{name}_val_accuracy'] = pd.to_numeric(results_df[f'final_{name}_val_accuracy'], errors='coerce')
    results_df[f'max_{name}_val_accuracy'] = pd.to_numeric(results_df[f'max_{name}_val_accuracy'], errors='coerce')

    # Filter runs with max teacher_val_accuracy above 83.0
    filtered_results = results_df[results_df[f'max_{name}_val_accuracy'] > 83.0]

    # Sort by max_teacher_val_accuracy in descending order
    filtered_results = filtered_results.sort_values(by=f'max_{name}_val_accuracy', ascending=False)

    # Display runs that achieved high teacher_val_accuracy during training
    print(f"Runs that achieved {name}_val_accuracy > 83.0 during training:")
    # print(filtered_results)

    # Find runs where max_teacher_val_accuracy is significantly higher than final_teacher_val_accuracy
    # This indicates runs that peaked high but ended lower
    results_df['accuracy_drop'] = results_df[f'max_{name}_val_accuracy'] - results_df[f'final_{name}_val_accuracy']
    peaked_runs = results_df[results_df['accuracy_drop'] > 0.5].sort_values(by='accuracy_drop', ascending=False)

    print(f"\nRuns with significant drop in {name}_val_accuracy (peaked high but ended lower):")
    print(peaked_runs[['run_id', 'run_name', f'max_{name}_val_accuracy', f'final_{name}_val_accuracy', 'accuracy_drop', 'max_accuracy_step', 'total_steps']])

    # If you want details on the run with the highest max teacher_val_accuracy
    if len(filtered_results) > 0:
        top_run_id = filtered_results.iloc[0]['run_id']
        print(f"\nFetching details for run with highest max {name}_val_accuracy: {top_run_id}")
        run = api.run(f"{project_path}/{top_run_id}")
        print(f"Run name: {run.name}")
        print(f"Max {name}_val_accuracy: {filtered_results.iloc[0][f'max_{name}_val_accuracy']}")
        print(f"Final {name}_val_accuracy: {filtered_results.iloc[0][f'final_{name}_val_accuracy']}")
        print(f"Occurred at step: {filtered_results.iloc[0]['max_accuracy_step']} of {filtered_results.iloc[0]['total_steps']}")
        print("Config:", run.config)
        update_yaml_config(name, run.config)  # Updates "distill_parameter" in config.yaml
    
# Restore standard output back to normal (optional)
sys.stdout = sys.__stdout__
