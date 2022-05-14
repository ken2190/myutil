import pyspark.sql.functions as f
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import *
from dq_data_cleaner.entities.address import Address
import pandas
from sparkaid import flatten
import json


ADDR_CLEAN_RESULT_SCHEMA = StructType([
    StructField('spk_addr_id', StringType(), True),
    StructField('spk_addr_id_is_valid', StringType(), True),
    StructField('spk_building_id', StringType(), True),
    StructField('spk_building_id_is_valid', StringType(), True),
    StructField('addr_result_code', StringType(), True),
    StructField('addr_input_is_modified', StringType(), True),
    StructField('addr_pre_processed', StringType(), True),
    StructField('addr_is_parsable', StringType(), True),
    StructField('addr_is_valid', StringType(), True),
    StructField('addr_full', StringType(), True),
    StructField('addr_house_num', StringType(), True),
    StructField('addr_pre_direction', StringType(), True),
    StructField('addr_street_name', StringType(), True),
    StructField('addr_suffix', StringType(), True),
    StructField('addr_post_direction', StringType(), True),
    StructField('addr_suite', StringType(), True),
    StructField('addr_suite_name', StringType(), True),
    StructField('addr_suite_num', StringType(), True),
    StructField('addr_city_postal', StringType(), True),
    StructField('addr_state_code', StringType(), True),
    StructField('addr_zip', StringType(), True),
    StructField('addr_zip_plus4', StringType(), True),
    StructField('addr_carrier_route', StringType(), True),
    StructField('addr_delivery_point_code', StringType(), True),
    StructField('addr_delivery_check_digit', StringType(), True),
    StructField('addr_dpv_footnote', StringType(), True),
    StructField('addr_type_string', StringType(), True),
    StructField('addr_county_name', StringType(), True),
    StructField('addr_county_fips', StringType(), True),
    StructField('addr_country_code', StringType(), True),
    StructField('addr_congressional_district', StringType(), True),
    StructField('addr_time_zone', StringType(), True),
    StructField('addr_time_zone_code', StringType(), True),
    StructField('addr_urbanization', StringType(), True),
    StructField('addr_zip_type', StringType(), True),
    StructField('addr_parsed_garbage', StringType(), True),
    StructField('addr_parsed_private_mailbox_name', StringType(), True),
    StructField('addr_parsed_private_mailbox_num', StringType(), True),
    StructField('addr_lacs', StringType(), True),
    StructField('addr_lacs_link_indicator', StringType(), True),
    StructField('addr_lacs_link_return_code', StringType(), True),
    StructField('addr_suite_link_return_code', StringType(), True),
    StructField('addr_rbdi', StringType(), True),
    StructField('addr_persistent_key', StringType(), True),
    StructField('addr_geo_result_code', StringType(), True),
    StructField('geo_latitude', StringType(), True),
    StructField('geo_longitude', StringType(), True),
    StructField('geo_census_tract', StringType(), True),
    StructField('geo_census_block', StringType(), True),
    StructField('geo_place_code', StringType(), True),
    StructField('geo_place_name', StringType(), True),
    StructField('geo_cbsa_code', StringType(), True),
    StructField('geo_cbsa_title', StringType(), True),
    StructField('geo_cbsa_level', StringType(), True),
    StructField('geo_cbsa_division_code', StringType(), True),
    StructField('geo_cbsa_division_title', StringType(), True),
    StructField('geo_cbsa_division_level', StringType(), True),
    StructField('geo_census_key', StringType(), True),
    StructField('geo_county_subdivision_code', StringType(), True),
    StructField('geo_county_subdivision_name', StringType(), True),
    StructField('geo_elementary_school_district_code', StringType(), True),
    StructField('geo_elementary_school_district_name', StringType(), True),
    StructField('geo_secondary_school_district_code', StringType(), True),
    StructField('geo_secondary_school_district_name', StringType(), True),
    StructField('geo_state_district_lower', StringType(), True),
    StructField('geo_state_district_upper', StringType(), True),
    StructField('geo_unified_school_district_code', StringType(), True),
    StructField('geo_unified_school_district_name', StringType(), True),
    StructField('geo_block_suffix', StringType(), True),
    StructField('geo_source_level', StringType(), True),
    StructField('geo_point_id', StringType(), True),
    StructField('geo_place_id', StringType(), True),
    StructField('geo_locale_id', StringType(), True),
    StructField('geo_nearby_locales', ArrayType(StringType(), True), True)]
    )


def preprocess_address_input(df):
    """
    Prepare input address data to feed into Address Cleaning: Back-fill Full Address Data
    """
    df_columns_list = df.columns
    address_lines_list = ["raw_addr_line1", "raw_addr_line2"]
    parsed_address_parts_list = ["raw_addr_house_num", "raw_addr_pre_dir", "raw_addr_street_name",
                                 "raw_addr_suffix", "raw_addr_unit_type", "raw_addr_unit_num"]
    concat_address_lines = f.concat_ws(' ', address_lines_list)
    concat_parsed_addresses = f.concat_ws(' ', parsed_address_parts_list)
    if "raw_addr_full" in df_columns_list:
        if address_lines_list in df_columns_list:
            df = df.withColumn("address_feed", f.when((f.col("raw_addr_full").isNull()), concat_address_lines)
                                 .otherwise(f.col("raw_addr_full")))
        elif parsed_address_parts_list in df_columns_list:
            df = df.withColumn("address_feed", f.when((f.col("raw_addr_full").isNull()), concat_parsed_addresses)
                                 .otherwise(f.col("raw_addr_full")))
        else:
            df = df.withColumn("address_feed", f.col("raw_addr_full"))
    elif address_lines_list in df_columns_list:
        df = df.withColumn("address_feed", concat_address_lines)
    elif parsed_address_parts_list in df_columns_list:
        df = df.withColumn("address_feed", concat_parsed_addresses)
    else:
        raise Exception("No Address Data is Available, "
                        "Please Make Sure Raw Address Field Names have been mapped to Spokeo's Master Schema Names")

    return df.withColumn("address_json_feed", f.to_json(f.struct(*[f.col('address_feed'),
                                                     f.col('raw_addr_city').alias('city'),
                                                     f.col('raw_addr_state').alias('state'),
                                                     f.col('raw_addr_zip').alias('zip')]))).drop("address_feed")


@pandas_udf(StringType())
def address_clean_udf(addr_json):
    """Pandas UDF to return each info as a dict."""
    result = []
    for index, value in addr_json.iteritems():
        address_input = json.loads(value)
        addr_obj = Address(address_input)
        result.append(addr_obj.to_json())
    return pandas.Series(result)


def address_clean(df):
    """
    Create address_json_feed & Run through Address Cleaning to get address cleaning results
    """
    df = df.withColumn("address_json_feed", f.to_json(f.struct(*[f.col('std_addr_full').alias('address'),
                                                            f.col('raw_addr_city').alias('city'),
                                                            f.col('raw_addr_state').alias('state'),
                                                            f.col('raw_addr_zip').alias('zip')]))).drop("std_addr_full")

    return df.withColumn("addr_clean_results", address_clean_udf("address_json_feed"))


def map_address_clean_result(df, result_column_name):
    """
    Map address cleaning results (in json string format) to individual columns
    """
    df = df.withColumn('clean', f.from_json(f.col(result_column_name), schema=ADDR_CLEAN_RESULT_SCHEMA))
    df = flatten(df).drop('clean')
    for column_name in df.columns:
        if column_name.startwith('clean_'):
            new_columne_name = column_name.lstrip('clean_')
            df = df.withColumnRenamed(column_name, new_columne_name)
