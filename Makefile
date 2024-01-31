BUILD := data/generated
SHELL := /bin/bash

all: report
experiments: experiment-1 experiment-2 experiment-3
experiments-long: experiment-1 experiment-2 experiment-3 experiment-5

######## Dataset Manager commands ########
status:
	python3 dataset_manager.py $@

######## Dataset Report Building ########
report: latex/report.pdf
latex/report.pdf: latex/report.tex data/source/dataset.csv $(BUILD)/report_requirements.stamp
	python3 report_plots.py
	cd latex && xelatex -8bit -shell-escape report && xelatex -8bit -shell-escape report

report_requirements: $(BUILD)/report_requirements.stamp
$(BUILD)/report_requirements.stamp: $(BUILD)/strictly-filtered-n1.stamp $(BUILD)/strictly-filtered-n2.stamp $(BUILD)/strictly-filtered-n3.stamp \
								  	experiments-long $(BUILD)/rule-training.stamp
	touch $@

test_distortion_resilience: $(BUILD)/kfold-validation-ds1-dr.stamp $(BUILD)/kfold-validation-ds2-dr.stamp $(BUILD)/function-tester-dr.stamp

######## Generating Synthetic Datasets ########
synthetic_dataset_current:
	python3 dataset_synthesizer.py config/experiments.yml DATASET_GENERATOR,CURRENT 2000

fully_synthetic_dataset:
	python3 dataset_synthesizer.py config/experiments.yml DATASET_GENERATOR,CURRENT 2000 --full-synth

######## Experiments ########
# Experiment 1
experiment-1: $(BUILD)/experiment-1.stamp
$(BUILD)/experiment-1.stamp: $(BUILD)/basic.stamp $(BUILD)/reference.stamp $(BUILD)/balanced.stamp
	touch $@

# Experiment 2
experiment-2: $(BUILD)/experiment-2.stamp
$(BUILD)/experiment-2.stamp: $(BUILD)/combiners.stamp
	touch $@

# Experiment 3
experiment-3: $(BUILD)/experiment-3.stamp
$(BUILD)/experiment-3.stamp: $(BUILD)/kfold-validation-ds1.stamp $(BUILD)/kfold-validation-ds2.stamp
	touch $@

# Experiment 5
experiment-5: $(BUILD)/experiment-5.stamp
$(BUILD)/experiment-5.stamp: $(BUILD)/function-tester.stamp
	touch $@


######## Experiment module details ########
$(BUILD)/basic.stamp: data/source/dataset.csv
	python3 basic_eval.py config/experiments.yml BASIC,INFORMED
	touch $@

$(BUILD)/reference.stamp: $(BUILD)/basic.stamp
	python3 reference.py config/experiments.yml REFERENCE
	touch $@

$(BUILD)/balanced.stamp: $(BUILD)/basic.stamp $(BUILD)/reference.stamp
	python3 sanitizer.py config/experiments.yml SANITIZER,INFORMED -b
	touch $@

$(BUILD)/strictly-filtered-n%.stamp: $(BUILD)/balanced.stamp
	python3 filter.py config/experiments.yml FILTER,N$* $* -s
	touch $@

$(BUILD)/combiners.stamp: $(BUILD)/balanced.stamp
	python3 combiners.py config/experiments.yml COMBINERS
	touch $@

$(BUILD)/rule-training.stamp: $(BUILD)/function-tester.stamp $(BUILD)/kfold-validation-ds1.stamp $(BUILD)/kfold-validation-ds2.stamp
	python3 rule_training.py config/experiments.yml RULE_TRAINING --discard-untrained --detect-best
	touch $@

$(BUILD)/kfold-validation-ds1.stamp: $(BUILD)/balanced.stamp
	python3 rule_optimization_kfold.py config/experiments.yml KFOLD_VALIDATION --method=distribute-confusion --also-run priority majority oracle dt rf ab --out=ds1
	touch $@

$(BUILD)/kfold-validation-ds2.stamp: $(BUILD)/balanced.stamp
	python3 rule_optimization_kfold.py config/experiments.yml KFOLD_VALIDATION --method=two-focal --out=ds2
	touch $@

$(BUILD)/param-optimization-complete.stamp: $(BUILD)/balanced.stamp
	python3 param_optimization.py config/experiments.yml PARAM_OPTIMIZATION --reduce 2000
	touch $@

$(BUILD)/param-optimization.stamp:  $(BUILD)/param-optimization-complete.stamp
	touch $@

$(BUILD)/function-tester.stamp: $(BUILD)/param-optimization.stamp $(BUILD)/balanced.stamp
	python3 function_param_tester.py config/experiments.yml FUNCTION_TESTER
	touch $@

######## Distortion Resilience Testing ########

$(BUILD)/kfold-validation-ds1-dr.stamp:
	python3 rule_optimization_kfold.py config/experiments.yml KFOLD_DISTORTION_RESILIENCE --distortion-resilience --method=distribute-confusion --also-run priority majority oracle dt rf ab --out=ds1
	touch $@

$(BUILD)/kfold-validation-ds2-dr.stamp:
	python3 rule_optimization_kfold.py config/experiments.yml KFOLD_DISTORTION_RESILIENCE --distortion-resilience --method=two-focal --out=ds2
	touch $@

$(BUILD)/function-tester-dr.stamp:
	python3 function_param_tester.py config/experiments.yml FUNCTION_TESTER_DISTORTION_RESILIENCE --distortion-resilience
	touch $@

######## Phony targets ########

.PHONY: clean clean-all all experiments experiments-long report_requirements \
		experiment-1 experiment-2 experiment-3 experiment-4 experiment-5 \
		status init init-synth

init:
	mkdir -p data/stash
	rm -f config/experiments.yml config/attr_conf
	cd config && ln -s attr_conf_real attr_conf
	cd config && ln -s experiments_classic.yml experiments.yml

init-synth:
	rm -f config/experiments.yml config/attr_conf
	cd config && ln -s attr_conf_synth attr_conf
	cd config && ln -s experiments_synth.yml experiments.yml

clean:
	rm -rf $(BUILD)/*
	rm -f latex/report.aux latex/report.log latex/report.out latex/report.toc

clean-all: clean
	rm -f latex/generated/* latex/report.pdf
