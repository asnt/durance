# Training data analysis

Generate statistics and graphical reports of training data for analysis.

# Goals

- Focus on endurance training.
- Focus on essential trends and statistics.
- Handle heartrate variation (HRV).
- Provide a web interface.

# Install

```bash
$ conda create -n durance \
    --file requirements/lib-conda.txt \
    --file requirements/web.txt
$ conda activate durance
(durance) $ pip install -r requirements/lib-pip.txt
```

# Run

```bash
# (Once)
(durance) $ python -m app.cli init
# (Regularly) Add activity recordings (.fit) to database.
(durance) $ python -m app.cli import path/to/recordings/*.fit
# Start web app (see localhost:8080).
(durance) $ python -m app.web
```

# License

GNU GPL v3

# See also

- [ActivityLog2](https://github.com/alex-hhh/ActivityLog2)
  Endurance and triathlon activity analysis.
  Exhaustive feature set.
  Desktop.
