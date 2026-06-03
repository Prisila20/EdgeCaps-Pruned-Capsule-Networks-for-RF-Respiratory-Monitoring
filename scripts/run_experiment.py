from src.models.teacher_capsule_model import build_optimized_model
from src.models.student_capsule_model import build_ultra_small_student
from src.data.loader import load_dataset
from src.training.train_models import train_model
from src.training.distillation import train_student_with_distillation
from src.evaluation.evaluate import evaluate_and_log
from src.utils import count_parameters, model_size_mb, compute_flops
from src.pruning.pruning_pipeline import prune_model
from src.models.baseline_models import get_all_baselines
import torch
import copy
import os
import argparse
torch.manual_seed(0)

def main(args):

    results = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = load_dataset(args.data_path, args.batch_size)


    # 1. TEACHER
    teacher_model = build_optimized_model(input_channels=3, image_height=64, image_width=64, num_classes=3, routings=3).to(device)

    teacher_model = train_model(teacher_model, train_loader, val_loader, device, args, prefix='teacher')
    torch.save(teacher_model.state_dict(), "teacher_model.pth")

    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    teacher_params = count_parameters(teacher_model)
    teacher_size = model_size_mb(teacher_model)
    teacher_flops = compute_flops(teacher_model)

    results.append(
        evaluate_and_log(teacher_model, "teacher", test_loader, device,
                        teacher_params, teacher_size, teacher_flops)
    )


    # 2. STUDENT (scratch)
    print("\nTraining student model from scratch")
    student = build_ultra_small_student(input_channels=3, image_height=64, image_width=64, num_classes=3, routings=3).to(device)
    student_trained = train_model(student, train_loader, test_loader, device, args, prefix='student_scratch')

    results.append(
        evaluate_and_log(student_trained, "student_scratch", test_loader, device,
                        teacher_params, teacher_size, teacher_flops)
    )


    # 3. PRUNED STUDENT
    print("\n Pruning student model")
    pruned = copy.deepcopy(student_trained)

    # prune
    pruned_model = prune_model(pruned, args.prune_ratio)

    # fine-tune
    pruned_trained = train_model(pruned_model, train_loader, val_loader, device, args, prefix='student_pruned')

    results.append(
        evaluate_and_log(pruned_trained, "student_pruned", test_loader, device,
                        teacher_params, teacher_size, teacher_flops)
    )

    # 4. KD STUDENT (no pruning)
    print("\n Training student model with KD")
    student_kd = copy.deepcopy(student)
    optimizer = torch.optim.Adam(student_kd.parameters(), lr=args.lr)

    student_kd = train_student_with_distillation(
        teacher_model, student_kd, train_loader, val_loader, optimizer, args, device=device, prefix='studen_kd'
    )

    results.append(
        evaluate_and_log(student_kd, "student_kd", test_loader, device,
                        teacher_params, teacher_size, teacher_flops)
    )


    # 5. PRUNED + KD
    print("\n Improving pruned model with KD")
    pruned_kd = copy.deepcopy(pruned_model)
    optimizer = torch.optim.Adam(pruned_kd.parameters(), lr=args.lr)

    pruned_kd = train_student_with_distillation(
        teacher_model, pruned_kd, train_loader, val_loader, optimizer, args,  device=device, prefix='student_pruned_kd'
    )

    results.append(
        evaluate_and_log(pruned_kd, "student_pruned_kd", test_loader, device,
                        teacher_params, teacher_size, teacher_flops)
    )


    # BASELINES
    print("\nTraining state of arts models")
    baselines = get_all_baselines()

    for name, model in baselines.items():
        print(f"\nTraining {name}")
        model = model.to(device)
        model = train_model(model, train_loader, test_loader, device, args, prefix=name)
        results.append(
            evaluate_and_log(model, name, test_loader, device,
                            teacher_params, teacher_size, teacher_flops)
        )


    df = pd.DataFrame(results)
    os.makedirs("csv_4", exist_ok=True)
    filename = f"csv_4/model_results_job_{args.job_id}_alpha_{args.alpha}_temp_{args.temperature}_ratio_{args.prune_ratio}_lr_{args.lr}.csv"

    df.to_csv(filename, index=False)

    print(df)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--prune_ratio", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--data_path", type=str, default="./data/EdgeCaps_datasets")
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument('--job_id', type=str, default=str(os.getpid()))

    args = parser.parse_args()
    main(args)