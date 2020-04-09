.PHONY: install

install:
	pip install .

.PHONY: profile

profile:
	awk '$$1=$$1' FS=";" OFS="," data/demographic_data_customers.csv > data/demographic_data_customers_temp.csv
	awk '$$1=$$1' FS=";" OFS="," data/demographic_data_german_population.csv > data/demographic_data_german_population_temp.csv
	
	pandas_profiling --title report-customers.html --minimal data/demographic_data_customers.csv.temp reports/report-customers.html
	pandas_profiling --title report-german-population.html --minimal data/demographic_data_german_population.csv.temp reports/report-german-population.html

	rm data/demographic_data_customers_temp.csv
	rm data/demographic_data_german_population_temp.csv

