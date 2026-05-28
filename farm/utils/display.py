"""Display helpers — work in both notebooks and terminals."""

import logging

logger = logging.getLogger("farm")


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def banner(text: str, emoji: str = "🔹") -> None:
    """Section header."""
    if _in_notebook():
        from IPython.display import display, HTML
        display(HTML(
            f'<div style="background:#1a1a2e;color:#e0e0ff;padding:10px 15px;'
            f'border-radius:8px;margin:10px 0;font-size:14px;font-family:monospace">'
            f'{emoji} <b>{text}</b></div>'
        ))
    else:
        logger.info(f"{emoji} {text}")


def ok(text: str) -> None:
    if _in_notebook():
        from IPython.display import display, HTML
        display(HTML(f'<span style="color:#4CAF50;font-weight:bold">✅ {text}</span>'))
    else:
        logger.info(f"✅ {text}")


def warn(text: str) -> None:
    if _in_notebook():
        from IPython.display import display, HTML
        display(HTML(f'<span style="color:#FF9800;font-weight:bold">⚠️ {text}</span>'))
    else:
        logger.warning(f"⚠️ {text}")


def fail(text: str) -> None:
    if _in_notebook():
        from IPython.display import display, HTML
        display(HTML(f'<span style="color:#f44336;font-weight:bold">❌ {text}</span>'))
    else:
        logger.error(f"❌ {text}")


def info_table(rows: list) -> None:
    """Display a key/value table.

    Parameters
    ----------
    rows : list of (str, str)
        Each element is a ``(label, value)`` pair.
    """
    if _in_notebook():
        from IPython.display import display, HTML
        html = '<table style="border-collapse:collapse;font-family:monospace;font-size:12px">'
        for key, val in rows:
            html += (
                f'<tr><td style="padding:3px 12px 3px 0;color:#aaa">{key}</td>'
                f'<td style="padding:3px 0;color:#fff"><b>{val}</b></td></tr>'
            )
        html += "</table>"
        display(HTML(
            f'<div style="background:#1e1e2e;padding:8px 12px;'
            f'border-radius:6px;margin:5px 0">{html}</div>'
        ))
    else:
        width = max((len(k) for k, _ in rows), default=0)
        for key, val in rows:
            logger.info(f"  {key:<{width}}  {val}")