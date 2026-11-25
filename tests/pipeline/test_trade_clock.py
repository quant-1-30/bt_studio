#! /usr/bin/env python3

import datetime

from bt_studio.trade import intraday_triggers, next_trigger


def test_intraday_triggers_match_backtest_timers():
    cc = {
        "bars_per_day": 240,
        "exclude_bars": 10,
        "T1_rets": {"exit_5m": 5, "exit_15m": 15, "exit_30m": 30},
    }
    triggers = intraday_triggers(cc)
    assert triggers == [
        (datetime.time(9, 35), "exit:5"),
        (datetime.time(9, 45), "exit:15"),
        (datetime.time(10, 0), "exit:30"),
        (datetime.time(14, 51), "entry"),
    ]


def test_next_trigger_ordering_and_end_of_day():
    cc = {"bars_per_day": 240, "exclude_bars": 10, "T1_rets": {"exit_5m": 5}}
    triggers = intraday_triggers(cc)  # [(09:35 exit:5), (14:51 entry)]

    day = datetime.date(2026, 11, 19)
    # before open -> first exit trigger
    dt, kind = next_trigger(triggers, datetime.datetime.combine(day, datetime.time(9, 0)))
    assert (dt.time(), kind) == (datetime.time(9, 35), "exit:5")
    # after the exit, before close -> entry trigger
    dt, kind = next_trigger(triggers, datetime.datetime.combine(day, datetime.time(10, 0)))
    assert (dt.time(), kind) == (datetime.time(14, 51), "entry")
    # after the entry -> nothing left today
    assert next_trigger(triggers, datetime.datetime.combine(day, datetime.time(14, 52))) is None
    # exact trigger instant is treated as already-fired (strictly-after)
    assert next_trigger(triggers, datetime.datetime.combine(day, datetime.time(9, 35))) is not None
