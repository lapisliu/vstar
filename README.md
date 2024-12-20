# *V*\*: Leveraging Multimodal AI for Traffic Light Detection: Evaluating SEAL and GPT-4V on Far Distance Traffic Signals

This repo is forked from [original vstar repo](https://github.com/penghao-wu/vstar).

### Installation
Running instructions are provided in the [greene-instruction.md](greene-instruction.md) file.

### Benchmarks
The benchmark files are in the [bench](bench) folder.

### Evaluation
`vstar_bench_eval.py` is the evaluation script. The evaluation results are saved in the `eval_result_[dataset_name].json` file. Change the last line in the main function from `eval_model(args)` to `eval_model_gpt4(args)` to evaluate the results with GPT-4V-mini for VQA. You need to export the OPENAI_API_KEY in the environment variables to use the GPT-4V-mini model.

`yolo_gpt_bench.py` is the evaluation script for YOLO in combination with GPT-4V-mini. Change the `benchmark_folder` to a specific benchmark folder to evaluate the results. The evaluation results are saved in `[benchmark_folder]_yolo_bench.json` file.