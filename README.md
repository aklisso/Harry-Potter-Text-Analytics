Harry Potter Text Analytics
The Harry Potter series is well-known and well-loved throughout the world, but readers often find some books more interesting than others. Through this project, my team and I sought to quantify variation in sentiment throughout the series by tracking valence and arousal, and their relationship with character presence. We:

- Cleaned and tokenized over 200 chapters of raw text from the Harry Potter series with Python NLTK, using regular expressions to segment text by chapter
- Calculated valence and arousal scores for each chapter using a customized Python package that integrates sentiment for domain-specific terms (i.e. "Voldemort")
- Tracked character presence throughout the series using word clouds, capturing character aliases with regular expressions
- Communicated findings through an interactive dashboard using Python Plotly and Dash

In order to access the dashboard on your own machine:

- Clone this GitHub repository
- Make sure you have following packages installed: polars, plotly , dash, dash_bootstrap_components, dash_bootstrap_templates
- In your command line, navigate to the directory with the repository and run "conda run -n python hp_sent_dash.py"
- Go to http://127.0.0.1:8050/ to view the dashboard
