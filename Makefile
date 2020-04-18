.PHONY: install

install:
	pip install .

.PHONY: test

test:
	pytest --pyargs customer_segmentation

.PHONY: profile

profile:
	awk '$$1=$$1' FS=";" OFS="," data/demographic_data_customers_sample.csv > data/demographic_data_customers_temp.csv
	awk '$$1=$$1' FS=";" OFS="," data/demographic_data_german_population_sample.csv > data/demographic_data_german_population_temp.csv
	
	pandas_profiling --title report-customers.html --minimal data/demographic_data_customers_temp.csv reports/report-customers.html
	pandas_profiling --title report-german-population.html --minimal data/demographic_data_german_population_temp.csv reports/report-german-population.html

	rm data/demographic_data_customers_temp.csv
	rm data/demographic_data_german_population_temp.csv

proposal.pdf:
	rst2pdf --stylesheets=twelvepoint proposal.rst proposal.pdf
