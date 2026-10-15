# atr-client

The Python client for this gateway, shared by its consumers
(`thodel/outremer`, `thodel/agentic_historian`). It lives here because the
gateway owns the contract: a route change ships with the client change, and
`tests/test_contract.py` runs the client against the real app.

```python
from atr_client import AtrClient

with AtrClient.from_env() as gw:          # ATR_GATEWAY_URL, ATR_API_KEY, ATR_HTTP_TIMEOUT
    ids = gw.list_model_ids()
    res = gw.read(page_png, model=ids[0], engine="kraken")   # /ocr or /recognize, by engine
    print(res.text, res.confidence, res.timing_ms)
```

Consumers pin a commit of this repository (no PyPI; pip honours
`#subdirectory` for VCS URLs only — an archive URL ignores it, measured with
pip 26.1):

```
atr-client @ git+https://github.com/thodel/serving-atr-inference@<40-hex sha>#subdirectory=clients/python
```

To change the client: edit it here, let this repository's CI pass, then bump
the SHA in the consumer that needs the change. Rationale and ownership of the
operational constants: outremer's
[ADR 0002](https://github.com/thodel/outremer/blob/main/docs/adr/0002-shared-code-with-agentic-historian.md).
