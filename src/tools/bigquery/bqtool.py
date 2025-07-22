import asyncio
import random
import sys
from typing import Any
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import enum
from typing import Any, Dict, Optional, TypeVar
from pydantic import BaseModel, ValidationError, validator

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions, JSONToolOutput

from google.cloud import bigquery
from google.oauth2 import service_account

from loguru import logger

load_dotenv()

project_id = os.getenv("search_project_id")
my_project_id = os.getenv("my_project_id")

credentials = service_account.Credentials.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

class BigQueryToolAction(str, enum.Enum):
    GetMetadata = "GET_METADATA"
    Query = "QUERY"

class BigQueryToolInput(BaseModel):
    action: BigQueryToolAction = Field(description="The action to perform BigQueryToolAction.GetMetadata get database tables structure, BigQueryToolAction.Query execute the Big Query query.")
    query: str = Field(default=None, description="The Big Query query to be executed, required for BigQueryToolAction.Query action.")

class BigQueryToolOutput(BaseModel):
    success: bool
    error: Optional[str] = None
    results: Optional[Any] = None

class BigQueryTool(Tool[BigQueryToolInput, ToolRunOptions, JSONToolOutput]):
    name = "World Bank BigQuery"
    description = f"""
    This tool enables users to query comprehensive World Bank datasets using natural language. \
    It supports exploration of global economic, health, education, debt, and demographic indicators. \
    Sense check your answers with your own general knowledge. \
    Provide retrieved data everytime to serve as objective evidence for your answers \
    Converts natural language to SQL query and executes it. IMPORTANT: strictly follow this order of actions:
    1. BigQueryToolAction.GetMetadata - get database tables structure (metadata)
    2. BigQueryToolAction.Query - execute the generated SQL query

    ALWAYS retrieve the metadata first.
    Queries MUST always be qualified with a project_id: {project_id}. Limit queries to 100000 tokens.

    Here are the tables in the form [dataset_id].[table_name], and their descriptions:
    - world_bank_intl_education.country_series_definitions: Contains country codes, series codes, and descriptions for education indicator series. Useful to discover what series codes correspond to which metrics.
        'country_code' is the ISO-3 country code, \
        'series_code' is the unique series identifier, \
        'description' is a text description of the series.

    - world_bank_intl_education.country_summary: Detailed country metadata including region, income group, currency, and various economic and survey-related fields. Use to filter or enrich country-level data.
        'country_code' is the ISO-3 country code, \
        'short_name' is the short country name, \
        'table_name' is an internal table reference, \
        'long_name' is the full country name, \
        'two_alpha_code' is the ISO-2 country code, \
        'currency_unit' is the currency used, \
        'special_notes' holds special notes about the country data, \
        'region' is the World Bank region grouping, \
        'income_group' classifies the country’s income level, \
        'wb_two_code' is the World Bank two-character code, \
        'national_accounts_base_year' and 'national_accounts_reference_year' refer to years for national accounts, \
        'sna_price_valuation' defines the System of National Accounts price valuation, \
        'lending_category' is the lending category, \
        'other_groups' includes other group classifications, \
        'system_of_national_accounts' defines the SNA version, \
        'alternative_conversion_factor' is a conversion factor, \
        'ppp_survey_year' is the year of the PPP survey, \
        'balance_of_payments_manual_in_use' shows manual status, \
        'external_debt_reporting_status' indicates debt reporting status, \
        'system_of_trade' defines trade classification system, \
        'government_accounting_concept' indicates accounting concepts, \
        'imf_data_dissemination_standard' is the IMF standard status, \
        'latest_population_census', 'latest_household_survey', 'source_of_most_recent_income_and_expenditure_data', 'vital_registration_complete', 'latest_agricultural_census' give years or status for various surveys, \
        'latest_industrial_data' and 'latest_trade_data' are years (integers) for industrial and trade data, \
        'latest_water_withdrawal_data' is the year for water withdrawal data.

    - world_bank_intl_education.international_education: Contains year-by-year education indicator values for countries. Use this table for querying actual indicator metrics.
        'country_name' is the full country name, \
        'country_code' is the ISO-3 code, \
        'indicator_name' is the name of the education indicator, \
        'indicator_code' is the indicator's shorthand code, \
        'value' is the numeric value of the indicator, \
        'year' is the observation year.

    - world_bank_intl_education.series_summary: Provides metadata about each education indicator series including definitions, measurement units, frequency, notes, and licensing. Use to understand indicator context.
        'series_code' is the series identifier, \
        'topic' is the category or topic of the series, \
        'indicator_name' is the indicator's full name, \
        'short_definition' and 'long_definition' explain the indicator, \
        'unit_of_measure' states the measurement unit, \
        'periodicity' is how often data is collected, \
        'base_period' is the base time period, \
        'other_notes', 'aggregation_method', 'limitations_and_exceptions', 'notes_from_original_source', 'general_comments' provide additional descriptive info, \
        'source' is the data source, \
        'statistical_concept_and_methodology' explains methodology, \
        'development_relevance' shows how the indicator relates to development goals, \
        'related_source_links' and 'other_web_links' provide URLs, \
        'related_indicators' lists related metrics, \
        'license_type' states the data license.

    - world_bank_wdi.country_series_definitions: Defines the mapping between countries and series codes along with descriptions. Useful to understand the meaning of series codes.
        'country_code' is the ISO-3 country code, \
        'series_code' is the unique series identifier, \
        'description' is a textual description of the series.

    - world_bank_wdi.country_summary: Contains detailed metadata for each country including codes, names, regions, economic groups, and various survey and accounting details. Useful for country-level filters and contextual data.
        'country_code' is a short geographic code for the country (ISO-3), \
        'short_name' is the country's conventional short name, \
        'table_name' is the official country name (conventional short form), \
        'long_name' is the official long form of the country name, \
        'two_alpha_code' is the ISO 2-digit country code, \
        'currency_unit' is the country's currency unit, \
        'special_notes' holds notes for data users, \
        'region' is the World Bank region classification, \
        'income_group' is the World Bank income group classification, \
        'wb_2_code' is the World Bank 2-digit country code, \
        'national_accounts_base_year' is the base year for national accounts pricing, \
        'national_accounts_reference_year' is the year to which constant price data are rescaled, \
        'sna_price_valuation' indicates valuation method for national accounts, \
        'lending_category' is the World Bank lending category, \
        'other_groups' includes other grouping types, \
        'system_of_national_accounts' shows which SNA version is used, \
        'alternative_conversion_factor' indicates if alternate currency conversion is used, \
        'ppp_survey_year' is the year of the latest PPP survey, \
        'balance_of_payments_manual_in_use' shows the balance of payments manual version, \
        'external_debt_reporting_status' indicates external debt reporting status, \
        'system_of_trade' shows the trade system classification, \
        'government_accounting_concept' shows accounting basis for government finance data, \
        'imf_data_dissemination_standard' indicates IMF data dissemination subscription, \
        'latest_population_census' year of the latest population census, \
        'latest_household_survey' details the latest demographic/education/health household surveys, \
        'source_of_most_recent_income_and_expenditure_data' is the source of income/expenditure data, \
        'vital_registration_complete' indicates if vital registration is complete, \
        'latest_agricultural_census' is the year of the latest agricultural census, \
        'latest_industrial_data' is the year for latest industrial data, \
        'latest_trade_data' is the year for the latest trade data.

    - world_bank_wdi.footnotes: Contains footnotes linked to country, series, and year data to provide clarifications or additional context.
        'country_code' is the ISO-3 country code, \
        'series_code' is the series identifier, \
        'year' is the year of the footnote, \
        'description' is the footnote text.

    - world_bank_wdi.indicators_data: Main table holding the indicator values per country and year. Use this for querying actual WDI metric data.
        'country_name' is the full country name, \
        'country_code' is the ISO-3 country code, \
        'indicator_name' is the descriptive name of the indicator, \
        'indicator_code' is the shorthand code for the indicator, \
        'value' is the numeric value of the indicator, \
        'year' is the observation year.

    - world_bank_wdi.series_summary: Contains detailed metadata about each indicator series including definitions, units, periodicity, notes, and data source info. Use to understand indicator context and limitations.
        'series_code' is the series identifier, \
        'topic' is the category or subject of the series, \
        'indicator_name' is the full indicator name, \
        'short_definition' is a brief explanation of the indicator, \
        'long_definition' is a detailed explanation, \
        'unit_of_measure' states measurement units, \
        'periodicity' defines data collection frequency, \
        'base_period' is the time period data is indexed to, \
        'other_notes' provides additional remarks, \
        'aggregation_method' describes how data are aggregated, \
        'limitations_and_exceptions' notes usage limitations, \
        'notes_from_original_source' provides original source comments, \
        'general_comments' holds other notes, \
        'source' identifies data source, \
        'statistical_concept_and_methodology' explains methodology, \
        'development_relevance' describes relevance to development goals, \
        'related_source_links' are URLs to related source info, \
        'other_web_links' are URLs to additional web resources, \
        'related_indicators' lists related indicator codes, \
        'license_type' states the data usage license.

    - world_bank_wdi.series_time: Provides time series information per series code, useful for querying which years have data for a series.
        'series_code' is the series identifier, \
        'year' is the year in the time series, \
        'description' is a note or description about the time period.    

    - world_bank_intl_debt.country_series_definitions: Contains country and series code definitions with descriptions. Use this table to understand the relationship between countries and series.
        'country_code' is the country code, \
        'series_code' is the series identifier, \
        'description' is the description of the series or country relationship. \

    - world_bank_intl_debt.country_summary: Contains metadata about countries including codes, names, classifications, and economic info. Use this table for country-level metadata.
        'country_code' is the country code, \
        'short_name' is the short country name, \
        'table_name' is the name used in tables, \
        'long_name' is the full official country name, \
        'two_alpha_code' is the ISO 2-letter code, \
        'currency_unit' is the currency used, \
        'special_notes' contains any special notes for the country, \
        'region' is the World Bank region, \
        'income_group' is the income classification group, \
        'wb_2_code' is the World Bank 2-digit code, \
        'national_accounts_base_year' is the base year for national accounts, \
        'national_accounts_reference_year' is the reference year for constant price calculations, \
        'sna_price_valuation' indicates basic or producer price valuation, \
        'lending_category' is the lending category, \
        'other_groups' are other groups assigned by WDI, \
        'system_of_national_accounts' shows which SNA system is used, \
        'alternative_conversion_factor' indicates use of alternative exchange rates, \
        'ppp_survey_year' is the latest PPP survey year, \
        'balance_of_payments_manual_in_use' shows which manual version is used, \
        'external_debt_reporting_status' indicates the quality/status of debt reporting, \
        'system_of_trade' shows trade system type, \
        'government_accounting_concept' indicates the government accounting basis, \
        'imf_data_dissemination_standard' shows IMF dissemination standards subscribed, \
        'latest_population_census' is the year of last population census, \
        'latest_household_survey' is the year of last household survey, \
        'source_of_most_recent_Income_and_expenditure_data' is the income/expenditure data source, \
        'vital_registration_complete' indicates completeness of vital stats registration, \
        'latest_agricultural_census' is the year of last agricultural census, \
        'latest_industrial_data' is the year of latest industrial data, \
        'latest_trade_data' is the year of latest trade data, \
        'latest_water_withdrawal_data' is the year of latest water withdrawal data. \

    - world_bank_intl_debt.international_debt: Contains year-by-year international debt indicator values for countries. Use this table for querying actual debt metrics.
        'country_name' is the full country name, \
        'country_code' is the ISO-3 country code, \
        'indicator_name' is the name of the debt indicator, \
        'indicator_code' is the shorthand indicator code, \
        'value' is the numeric value of the indicator, \
        'year' is the observation year. \

    - world_bank_intl_debt.series_summary: Contains metadata about series codes with detailed definitions and notes. Use this table to understand what each series measures.
        'series_code' is the series identifier, \
        'topic' is the topic category, \
        'indicator_name' is the indicator's name, \
        'short_definition' is a brief definition, \
        'long_definition' is a detailed definition, \
        'unit_of_measure' is the measurement unit, \
        'periodicity' is data frequency, \
        'base_period' is the base period for indices, \
        'other_notes' are additional notes, \
        'aggregation_method' shows how data is aggregated, \
        'limitations_and_exceptions' lists known limitations, \
        'notes_from_original_source' includes original source notes, \
        'general_comments' holds other comments, \
        'source' is the data source, \
        'statistical_concept_and_methodology' explains methodology, \
        'development_relevance' states relevance to development goals, \
        'related_source_links' are URLs to related sources, \
        'other_web_links' are other relevant URLs, \
        'related_indicators' are related series, \
        'license_type' details licensing info. \

    - world_bank_intl_debt.series_times: Contains yearly time series notes for each series code. Use this table for temporal descriptions or notes.
        'series_code' is the series identifier, \
        'year' is the year of the note, \
        'description' is the note or description for that year. \

    - world_bank_health_population.country_series_definitions: Contains country and series code definitions with descriptions. Use this table to understand the relationship between countries and series.
        'country_code' is the country code, \
        'series_code' is the series identifier, \
        'description' is the description of the series or country relationship. \

    - world_bank_health_population.country_summary: Contains metadata about countries including codes, names, classifications, and economic info. Use this table for country-level metadata.
        'country_code' is the country code, \
        'short_name' is the short country name, \
        'table_name' is the name used in tables, \
        'long_name' is the full official country name, \
        'two_alpha_code' is the ISO 2-letter code, \
        'currency_unit' is the currency used, \
        'special_notes' contains any special notes for the country, \
        'region' is the World Bank region, \
        'income_group' is the income classification group, \
        'wb_2_code' is the World Bank 2-digit code, \
        'national_accounts_base_year' is the base year for national accounts, \
        'national_accounts_reference_year' is the reference year for constant price calculations, \
        'sna_price_valuation' indicates basic or producer price valuation, \
        'lending_category' is the lending category, \
        'other_groups' are other groups assigned by WDI, \
        'system_of_national_accounts' shows which SNA system is used, \
        'alternative_conversion_factor' indicates use of alternative exchange rates, \
        'ppp_survey_year' is the latest PPP survey year, \
        'balance_of_payments_manual_in_use' shows which manual version is used, \
        'external_debt_reporting_status' indicates the quality/status of debt reporting, \
        'system_of_trade' shows trade system type, \
        'government_accounting_concept' indicates the government accounting basis, \
        'imf_data_dissemination_standard' shows IMF dissemination standards subscribed, \
        'latest_population_census' is the year of last population census, \
        'latest_household_survey' is the year of last household survey, \
        'source_of_most_recent_income_and_expenditure_data' is the income/expenditure data source, \
        'vital_registration_complete' indicates completeness of vital stats registration, \
        'latest_agricultural_census' is the year of last agricultural census, \
        'latest_industrial_data' is the year of latest industrial data, \
        'latest_trade_data' is the year of latest trade data. \

    - world_bank_health_population.health_nutrition_population: Contains year-by-year health, nutrition, and population indicator values for countries. Use this table for querying actual indicator metrics.
        'country_name' is the full country name, \
        'country_code' is the ISO-3 country code, \
        'indicator_name' is the name of the health or nutrition indicator, \
        'indicator_code' is the shorthand indicator code, \
        'value' is the numeric value of the indicator, \
        'year' is the observation year. \

    - world_bank_health_population.series_summary: Contains metadata about series codes with detailed definitions and notes. Use this table to understand what each series measures.
        'series_code' is the series identifier, \
        'topic' is the topic category, \
        'indicator_name' is the indicator's name, \
        'short_definition' is a brief definition, \
        'long_definition' is a detailed definition, \
        'unit_of_measure' is the measurement unit, \
        'periodicity' is data frequency, \
        'base_period' is the base period for indices, \
        'other_notes' are additional notes, \
        'aggregation_method' shows how data is aggregated, \
        'limitations_and_exceptions' lists known limitations, \
        'notes_from_original_source' includes original source notes, \
        'general_comments' holds other comments, \
        'source' is the data source, \
        'statistical_concept_and_methodology' explains methodology, \
        'development_relevance' states relevance to development goals, \
        'related_source_links' are URLs to related sources, \
        'other_web_links' are other relevant URLs, \
        'related_indicators' are related series, \
        'license_type' details licensing info. \

    - world_bank_health_population.series_times: Contains yearly time series notes for each series code. Use this table for temporal descriptions or notes. 
        'series_code' is the series identifier, \
        'year' is the year of the note, \
        'description' is the note or description for that year. \

    - world_bank_global_population.population_by_country: Contains yearly population counts by country from 1960 to 2019. Use this table for querying historical population data. 
        'country' is the full country name, \
        'country_code' is the ISO-3 country code, \
        'year_1960' to 'year_2019' columns represent the population count for each respective year as an integer. \

    """

    input_schema = BigQueryToolInput

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "example", "bigquery"],
            creator=self,
        )

    def validate_input(self, input_data: Dict[str, Any]):
        if input_data["action"] == BigQueryToolAction.Query and not input_data.get("query"):
            raise ValidationError("SQL Query is required for QUERY action.")

    def connection(self) -> bigquery.client.Client:
        try:
            engine = bigquery.Client(credentials=credentials, project=project_id)
            return engine
        except Exception as e:
            raise Exception(f"Unable to connect to database: {e}")

    def get_metadata(self, engine: bigquery.client.Client, project: str = my_project_id):
        metadata_query =f'''
        SELECT 'world_bank_global_population' AS dataset_name, table_name, column_name, data_type, ordinal_position
        FROM `bigquery-public-data.world_bank_global_population.INFORMATION_SCHEMA.COLUMNS`

        UNION ALL

        SELECT 'world_bank_health_population' AS dataset_name, table_name, column_name, data_type, ordinal_position
        FROM `bigquery-public-data.world_bank_health_population.INFORMATION_SCHEMA.COLUMNS`

        UNION ALL

        SELECT 'world_bank_intl_debt' AS dataset_name, table_name, column_name, data_type, ordinal_position
        FROM `bigquery-public-data.world_bank_intl_debt.INFORMATION_SCHEMA.COLUMNS`

        UNION ALL

        SELECT 'world_bank_intl_education' AS dataset_name, table_name, column_name, data_type, ordinal_position
        FROM `bigquery-public-data.world_bank_intl_education.INFORMATION_SCHEMA.COLUMNS`

        UNION ALL

        SELECT 'world_bank_wdi' AS dataset_name, table_name, column_name, data_type, ordinal_position
        FROM `bigquery-public-data.world_bank_wdi.INFORMATION_SCHEMA.COLUMNS`

        ORDER BY dataset_name, table_name, ordinal_position;
                '''
        job_config = bigquery.QueryJobConfig()
        
        metadata = engine.query_and_wait(metadata_query, job_config=job_config, project=project).to_dataframe()
        metadata = metadata.to_dict('records')

        # [
        # { "table_name": 'table1', "column_name": 'column1', "data_type": 'integer' },
        # { "table_name": 'table1', "column_name": 'column2', "data_type": 'varchar' },
        # { "table_name": 'table2', "column_name": 'column3', "data_type": 'boolean' },
        # { "table_name": 'table2', "column_name": 'column4', "data_type": 'date' },
        #   ]

        return metadata

    def execute_query(self, query: str, engine: bigquery.client.Client, project: str = my_project_id):
        if not self.is_read_only_query(query):
            return JSONToolOutput(
                    result={
                        "success": False,
                        "error": "Invalid query. Only SELECT queries are allowed."
                    }
                )
        try:
            # get result from engine(query)
            job_config = bigquery.QueryJobConfig()
            results = engine.query_and_wait(query, job_config=job_config, project=project).to_dataframe()
            results = results.to_dict('records')
            if len(results) > 0:
                return JSONToolOutput(
                    result={
                        "success": True,
                        "results": results
                    }
                )
            else:
                return JSONToolOutput(
                    result={
                        "success": False,
                        "error": "No rows selected"
                    }
                )
        except Exception as e:
            raise Exception(f"Generate a correct query that retrieves data using the appropriate dialect. The original request was: {query}, and the error was: {e}")

    def is_read_only_query(self, query: str):
        normalized_query = query.strip().upper()
        return normalized_query.startswith("SELECT") or normalized_query.startswith("SHOW") or normalized_query.startswith("DESC")

    async def _run(
        self, input: BigQueryToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput:  
        if input.action == BigQueryToolAction.GetMetadata:
            engine = self.connection()
            metadata = self.get_metadata(engine)
            return JSONToolOutput(
                    result={
                        "success": True,
                        "results": metadata
                    }
                )
        elif input.action == BigQueryToolAction.Query:
            engine = self.connection()
            return self.execute_query(input.query, engine)
        else:
            raise Exception("Invalid action specified")

