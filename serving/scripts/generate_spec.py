import json
from pathlib import Path

import api.app

with Path(
    f'{Path(__file__).resolve().parent}/../../openapi.spec.json'
).open('w') as f:
    app = api.app.setup()
    json.dump(app.openapi(), f, indent=4)