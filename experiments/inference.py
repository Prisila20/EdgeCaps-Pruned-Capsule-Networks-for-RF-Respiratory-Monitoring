import torch
import time
import matplotlib.pyplot as plt
import os
import numpy as np

def compare_model_inference(teacher_model, student_model, distilled_model, test_loader, num_runs=10):
    """
    Measures and compares inference times for Teacher, Student, and Distilled models with multiple runs per sample.
    
    Args:
        teacher_model (torch.nn.Module): The teacher model.
        student_model (torch.nn.Module): The student model.
        distilled_model (torch.nn.Module): The distilled model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test dataset.
        num_runs (int): Number of times each sample is run for averaging inference time.

    Returns:
        None (Displays a plot comparing inference times)
    """

    # Set models to evaluation mode
    teacher_model.eval()
    student_model.eval()
    distilled_model.eval()

    # # Detect device (Use GPU if available)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move models to device
    teacher_model.to(device)
    student_model.to(device)
    distilled_model.to(device)

    def measure_inference_time(model, test_loader, num_runs):
        inference_times = []
        total_samples = 0  # Counter for processed samples

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if isinstance(data, (list, tuple)):  
                    inputs = data[0].to(device)  # Unpack input tensor
                else:
                    inputs = data.to(device)  # Directly use if it's already a tensor
                
                # Run inference multiple times and take the average time
                total_time = 0
                for _ in range(num_runs):
                    start_time = time.time()  # Start timer
                    model(inputs)  # Run inference
                    end_time = time.time()  # End timer
                    total_time += (end_time - start_time)

                avg_time = (total_time / num_runs) * 1000  # Convert to milliseconds
                inference_times.append(avg_time)
                total_samples += 1  # Increment counter

        # print(f"Processed {total_samples} samples for {model.__class__.__name__}")
        return inference_times

    # Measure inference times with multiple runs
    teacher_times = measure_inference_time(teacher_model, test_loader, num_runs)
    student_times = measure_inference_time(student_model, test_loader, num_runs)
    distilled_times = measure_inference_time(distilled_model, test_loader, num_runs)
    
    avg_teacher_time = np.mean(teacher_times[1:])
    avg_student_time = np.mean(student_times[1:])
    avg_distilled_time = np.mean(distilled_times[1:])

    # Print average times
    print(f"\n**Average Inference Times:**")
    print(f"   - Teacher Model: {avg_teacher_time:.4f} ms")
    print(f"   - Student Model: {avg_student_time:.4f} ms")
    print(f"   - Distilled Model: {avg_distilled_time:.4f} ms\n")