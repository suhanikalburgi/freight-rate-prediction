# Truckload Price Prediction

I trained a gradient boosting model on 48,000 shipments to predict prices for 12,000 November and December 2025 shipments. The model estimates price per mile from the route, distance, truck type, weight and date. Multiplying by distance gives the total price in US dollars.

## Validation results

I trained on earlier months and tested on August, September and October separately. The average absolute error per shipment was $128.34 in August, $103.61 in September and $104.91 in October. The October baseline, which used the median price per mile for each truck type, had an error of $231.30.

## Reproduce the predictions

Tested with Python 3.12. The script reads the four assessment CSVs from data/ and saves both prediction files in outputs/. Run from the project folder.

    python -m pip install -r requirements.txt
    python freight_rates.py

To generate the December chart, run the supplied score.py from the same folder.

    python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_predictions.csv --output-dir outputs/scorer_results
