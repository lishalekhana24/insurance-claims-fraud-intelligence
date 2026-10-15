ThisBuild / scalaVersion := "2.12.18"
ThisBuild / version := "0.1.0-SNAPSHOT"

lazy val root = (project in file("."))
  .settings(
    name := "insurance-claims-fraud-intelligence",
    Compile / unmanagedSourceDirectories += baseDirectory.value / "src" / "spark",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % "3.5.3" % Provided,
      "org.apache.spark" %% "spark-sql-kafka-0-10" % "3.5.3" % Provided
    )
  )
