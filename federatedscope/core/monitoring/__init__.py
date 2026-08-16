"""Structured monitoring primitives shared by training and control layers."""

from federatedscope.core.monitoring.events import (
    EVENT_MARKER, emit_training_event, parse_training_event_line)

__all__ = ['EVENT_MARKER', 'emit_training_event',
           'parse_training_event_line']
