import asyncio
import random
import sys
from typing import Any
import os
from pydantic import BaseModel, Field
import enum
from typing import Any, Dict, Optional, TypeVar
from pydantic import BaseModel, ValidationError, validator

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions, JSONToolOutput
import datetime


class FiscalWeekToolInput(BaseModel):
    date: datetime.date = Field(default=None, description="The input date to be converted.")

class FiscalWeekTool(Tool[FiscalWeekToolInput, ToolRunOptions, JSONToolOutput]):
    name = "FiscalWeek"
    description = f"""
    Converts a date into a 'fiscal_week_id' object to use as input to the finance_operating_statement table. 'fiscal_week_id' is financial year week number in the format YYYYWW where YYYY is Australian financial year and WW is the financial year week number; for example 202501 corresponds to the first week of July 2024,

    """

    input_schema = FiscalWeekToolInput

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "example", "fiscalweek"],
            creator=self,
        )

    def to_fiscalweekid(self, d: datetime.date):
        fy =  d.year + 1 if d.month > 6 else d.year
        week = self.get_aus_financial_year_week(d)
        result = str(fy) + str(week)
        return JSONToolOutput(
                result={
                    "success": True,
                    "results": result
                }
            )

    def get_aus_financial_year_week(self, date):
        # Determine the start of the financial year
        if date.month >= 7:
            fy_start = datetime.date(date.year, 7, 1)
        else:
            fy_start = datetime.date(date.year - 1, 7, 1)
        
        # Calculate the difference in days
        delta_days = (date - fy_start).days
        # Week number: integer division by 7, plus 1 to make it 1-based
        week_number = delta_days // 7 + 1
        return f"{week_number:02d}"

    async def _run(
        self, input: FiscalWeekToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput:  
        return self.to_fiscalweekid(input.date)

