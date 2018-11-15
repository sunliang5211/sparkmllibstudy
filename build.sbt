import sun.security.tools.PathList

name := "sparkmllibstudy"

version := "1.0"

scalaVersion := "2.10.6"

resolvers += Resolver.mavenLocal
resolvers += "aliyun Maven Repository" at "http://maven.aliyun.com/nexus/content/groups/public"
externalResolvers := Resolver.withDefaultResolvers(resolvers.value,mavenCentral=false)

val spark_code = "org.apache.spark" %% "spark-core" % "1.6.1"
val spark_mllib = "org.apache.spark" %% "spark-mllib" % "1.6.1"

lazy val root = (project in file("."))
  .settings(
    libraryDependencies += spark_code,
    libraryDependencies += spark_mllib
  )