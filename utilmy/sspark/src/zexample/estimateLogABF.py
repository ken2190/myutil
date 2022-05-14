from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.linalg import VectorUDT, Vectors
import pyspark.sql.types as T
import pyspark.sql.functions as F

sparkConf = SparkConf()
sparkConf = sparkConf.set('spark.hadoop.fs.gs.requester.pays.mode', 'AUTO')
sparkConf = sparkConf.set('spark.hadoop.fs.gs.requester.pays.project.id',
                          'open-targets-eu-dev')

# establish spark connection
spark = (
    SparkSession.builder
    .config(conf=sparkConf)
    .master('local[*]')
    .getOrCreate()
)

# credSetPath = "gs://genetics-portal-dev-staging/finemapping/220228_merged/credset/"
# credSet = (
#     spark.read.json(credSetPath)
#     .distinct()
#     .withColumn("key", F.concat_ws('_', *['study_id', 'phenotype_id', 'bio_feature']))
# )

variantIndexPath = "gs://genetics-portal-dev-data/22.02.1/outputs/lut/variant-index"
variantIndex = spark.read.parquet(variantIndexPath)

credSet22Path = "gs://genetics-portal-dev-analysis/dsuveges/test_credible_set_chr22.parquet"
credSet = (
    spark.read.parquet(credSet22Path)
    .distinct()
    .withColumn("key", F.concat_ws('_', *['study_id', 'phenotype_id', 'bio_feature']))
)

@F.udf(returnType=T.FloatType())
def sdYest(vbeta: VectorUDT, maf: VectorUDT, n: T.FloatType()):
    oneover = 1/vbeta
    nvx = 2 * n * maf * (1-maf)
    # m <- lm(nvx ~ oneover - 1)
    # if(coef(m)[["oneover"]] < 0)
    #stop("Trying to estimate trait variance from betas, and getting negative estimate.  Something is wrong.  You can 'fix' this by supplying an estimate of trait standard deviation yourself, as sdY=<value> in the dataset list.")
    # return(sqrt(coef(m)[["oneover"]]))
    return(float(1))

    
sdY.est <- function(vbeta, maf, n) {
  oneover <- 1/vbeta
  nvx <- 2 * n * maf * (1-maf)
  m <- lm(nvx ~ oneover - 1)
  if(coef(m)[["oneover"]] < 0)
    stop("Trying to estimate trait variance from betas, and getting negative estimate.  Something is wrong.  You can 'fix' this by supplying an estimate of trait standard deviation yourself, as sdY=<value> in the dataset list.")
  return(sqrt(coef(m)[["oneover"]]))
}


## TODO: calculate logABF from data, because Finngen studies (Sussie results) don't have logABF
## https://github.com/tobyjohnson/gtx/blob/9afa9597a51d0ff44536bc5c8eddd901ab3e867c/R/abf.R#L53
traitSDYs = (
    credSet
    # Getting MAFs
    .join(
        variant
        .select(
            F.col("chr_id").alias("tag_chrom"),
            F.col("position").alias("tag_pos"),
            F.col("ref_allele").alias("tag_ref"),
            F.col("alt_allele").alias("tag_alt"),
            F.col("gnomad_fin").alias("tag_maf_fin"),
            F.col("gnomad_nfe").alias("tag_maf_nfe")
        ),
        on = ["tag_chrom", "tag_pos", "tag_ref", "tag_alt"],
        how = "left"
    )
    .withColumn("vbeta", F.col("tag_se_cond") ** 2)
    .withColumn("oneover", 1 / F.col("vbeta"))
    ## TODO: update sample size
    .withColumn("sample_size", F.lit(5000))
    ## TODO: update MAF
    .withColumn("maf", F.when(F.rand() > 0.5, 0.5).otherwise(0.25))
    .withColumn("nvx", 
                2 * F.col("sample_size") * F.col("maf") * (1 - F.col("maf")))
    ## TODO: consolidate with credSetLogSums?
)

assembler = (
    VectorAssembler()
    .setInputCols(["oneover"])
    .setOutputCol("features")
)
df = assembler.transform(traitSDYs.select("oneover", F.col("nvx").alias("label")))

lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    fitIntercept=False
)
lr_Model = lr.fit(df)

testSet = lr_Model.transform(df)


lr_Model.coefficients