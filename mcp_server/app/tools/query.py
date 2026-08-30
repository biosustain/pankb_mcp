"""
Database Query Tools for PanKB MCP Server

Data hierarchy: Family -> Species -> Genome -> Gene
"""
import json
import logging
from typing import List, Optional

import pandas as pd
from app.config import Config
from app.utils.connections import mongo_client
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(name="QueryTools")


def make_table_response(title: str, columns: list[str], rows: list[dict], summary: str = "") -> str:
    """Create a standardized table response in JSON format for frontend rendering"""
    response = {
        "type": "table",
        "title": title,
        "columns": columns,
        "rows": rows,
        "summary": summary,
        "row_count": len(rows)
    }
    return json.dumps(response)


@mcp.tool()
def list_families(family: Optional[str] = None) -> str:
    """
    List microbial families in PanKB.
    Returns family-level statistics including species count and total genomes.

    Args:
        family: Filter by family name (case-insensitive)
    """
    collection = mongo_client.get_collection(Config.MONGODB_COLLECTIONS["organisms"])

    match_stage = {}
    if family:
        match_stage["family"] = {"$regex": f"^{family}$", "$options": "i"}

    pipeline = [
        {"$match": match_stage} if match_stage else {"$match": {}},
        {
            "$group": {
                "_id": "$family",
                "species_count": {"$sum": 1},
                "total_genomes": {"$sum": "$genomes_num"},
                "total_genes": {"$sum": "$genes_num"}
            }
        },
        {"$sort": {"total_genomes": -1}}
    ]

    results = list(collection.aggregate(pipeline))

    if not results:
        return "No families found."

    rows = []
    for r in results:
        rows.append({
            "family": r["_id"],
            "species_count": r["species_count"],
            "total_genomes": r["total_genomes"],
            "total_genes": r["total_genes"]
        })

    return make_table_response(
        title="Microbial Families",
        columns=["family", "species_count", "total_genomes", "total_genes"],
        rows=rows,
        summary=f"Found {len(results)} families"
    )


@mcp.tool()
def list_species(
    family: Optional[str] = None,
    species: Optional[str] = None,
    pangenome_analysis: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """
    List species (pangenome analyses) in PanKB.
    Returns pangenome statistics including core/shell/cloud gene counts and openness.

    Args:
        family: Filter by family name
        species: Search by species name (case-insensitive)
        pangenome_analysis: Exact pangenome analysis name (e.g., 'Bacillus_subtilis')
        limit: Maximum number of results (default: None, returns all)
    """
    collection = mongo_client.get_collection(Config.MONGODB_COLLECTIONS["organisms"])

    query = {}
    if family:
        query["family"] = {"$regex": f"^{family}$", "$options": "i"}
    if species:
        # Normalize species name: replace spaces with underscores for regex search
        species_normalized = species.replace(" ", "_")
        query["species"] = {"$regex": f"^{species_normalized}$", "$options": "i"}
    if pangenome_analysis:
        # Normalize species name: replace spaces with underscores
        query["pangenome_analysis"] = pangenome_analysis.replace(" ", "_")

    cursor = collection.find(query, {"_id": 0})
    if limit:
        cursor = cursor.limit(limit)
    results = list(cursor)

    if not results:
        return "No species found matching the criteria."

    rows = []
    for org in results:
        gene_dist = org.get("gene_class_distribution", [0, 0, 0])
        if isinstance(gene_dist, list) and len(gene_dist) >= 3:
            core, shell, cloud = gene_dist[0], gene_dist[1], gene_dist[2]
        else:
            core, shell, cloud = 0, 0, 0

        rows.append({
            "species": org.get("species"),
            "family": org.get("family"),
            "genomes": org.get("genomes_num", 0),
            "genes": org.get("genes_num", 0),
            "core": core,
            "shell": shell,
            "cloud": cloud,
            "openness": org.get("openness", "N/A")
        })

    return make_table_response(
        title="Species (Pangenome Analyses)",
        columns=["species", "family", "genomes", "genes", "core", "shell", "cloud", "openness"],
        rows=rows,
        summary=f"Found {len(results)} species"
    )


@mcp.tool()
def list_genomes(
    pangenome_analysis: Optional[str] = None,
    genome_ids: Optional[List[str]] = None,
    country: Optional[str] = None,
    isolation_source: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """
    List genomes in PanKB.
    Returns genome details including GC content, length, phylogroup, and isolation info.

    Args:
        pangenome_analysis: Filter by pangenome analysis (species)
        genome_ids: List of specific genome IDs to query
        country: Filter by isolation country
        isolation_source: Filter by isolation source (e.g., 'Soil', 'Blood')
        limit: Maximum number of results (default: None, returns all)
    """
    collection = mongo_client.get_collection(Config.MONGODB_COLLECTIONS["genome_info"])

    match_query = {}
    if pangenome_analysis:
        # Normalize species name: replace spaces with underscores
        match_query["pangenome_analysis"] = pangenome_analysis.replace(" ", "_")
    if genome_ids:
        match_query["genome_id"] = {"$in": genome_ids}

    pipeline = [
        {"$match": match_query} if match_query else {"$match": {}},
        {
            "$lookup": {
                "from": Config.MONGODB_COLLECTIONS["isolation_info"],
                "localField": "genome_id",
                "foreignField": "genome_id",
                "as": "isolation"
            }
        },
        {"$unwind": {"path": "$isolation", "preserveNullAndEmptyArrays": True}},
    ]

    if country or isolation_source:
        isolation_filter = {}
        if country:
            isolation_filter["isolation.country"] = {"$regex": f"^{country}$", "$options": "i"}
        if isolation_source:
            isolation_filter["isolation.isolation_source"] = {"$regex": f"^{isolation_source}$", "$options": "i"}
        pipeline.append({"$match": isolation_filter})

    if limit:
        pipeline.append({"$limit": limit})

    results = list(collection.aggregate(pipeline))

    if not results:
        return "No genomes found matching the criteria."

    rows = []
    for doc in results:
        isolation = doc.get("isolation", {})

        rows.append({
            "genome_id": doc.get("genome_id"),
            "species": doc.get("species"),
            "strain": doc.get("strain", "N/A"),
            "gc_content": round(doc.get("gc_content", 0) * 100, 2),
            "genome_len": doc.get("genome_len", 0),
            "phylo_group": doc.get("phylo_group", "N/A"),
            "country": isolation.get("country", "N/A"),
            "isolation_source": isolation.get("isolation_source", "N/A")
        })

    return make_table_response(
        title="Genomes",
        columns=["genome_id", "species", "strain", "gc_content", "genome_len", "phylo_group", "country", "isolation_source"],
        rows=rows,
        summary=f"Found {len(results)} genomes"
    )


@mcp.tool()
def list_genes(
    gene_names: Optional[List[str]] = None,
    pangenome_analysis: Optional[str] = None,
    pangenomic_class: Optional[str] = None,
    cog_category: Optional[str] = None,
    protein_search: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """
    List genes in PanKB.
    Returns gene details including annotations, frequency, and pangenomic class.

    IMPORTANT: When searching by gene name without species filter, returns the gene
    across ALL species (same gene name exists in multiple species). Only add
    pangenome_analysis filter if user explicitly specifies a species.

    Args:
        gene_names: List of gene names to query (searches across all species by default)
        pangenome_analysis: Filter by species - only use if user explicitly requests a specific species
        pangenomic_class: Filter by pangenomic class ('Core', 'Accessory', or 'Rare')
        cog_category: Filter by COG category (e.g., 'K' for Transcription)
        protein_search: Search in protein descriptions (case-insensitive)
        limit: Maximum number of results (default: None, returns all)
    """
    collection = mongo_client.get_collection(Config.MONGODB_COLLECTIONS["gene_annotations"])

    query = {}
    if pangenome_analysis:
        # Normalize species name: replace spaces with underscores
        query["pangenome_analysis"] = pangenome_analysis.replace(" ", "_")
    if gene_names:
        query["gene"] = {"$in": gene_names}
    if pangenomic_class:
        query["pangenomic_class"] = pangenomic_class
    if cog_category:
        query["cog_category"] = cog_category
    if protein_search:
        query["protein"] = {"$regex": f"^{protein_search}$", "$options": "i"}

    cursor = collection.find(query, {"_id": 0})
    if limit:
        cursor = cursor.limit(limit)
    results = list(cursor)

    if not results:
        return "No genes found matching the criteria."

    rows = []
    for gene in results:
        protein = gene.get("protein", "N/A")
        rows.append({
            "gene": gene.get("gene"),
            "species": gene.get("species"),
            "protein": protein[:50] + "..." if len(protein) > 50 else protein,
            "pangenomic_class": gene.get("pangenomic_class"),
            "frequency": gene.get("frequency", 0),
            "cog_category": gene.get("cog_category", "-"),
            "cog_name": gene.get("cog_name", "N/A")
        })

    return make_table_response(
        title="Genes",
        columns=["gene", "species", "protein", "pangenomic_class", "frequency", "cog_category", "cog_name"],
        rows=rows,
        summary=f"Found {len(results)} genes"
    )


# @mcp.tool()
# def list_pathways(
#     pathway_ids: Optional[List[str]] = None,
#     pathway_name_search: Optional[str] = None,
#     limit: Optional[int] = None
# ) -> str:
#     """
#     List KEGG pathways in PanKB.

#     Args:
#         pathway_ids: KEGG pathway IDs (e.g., ['map00010', 'map00020'])
#         pathway_name_search: Search pathway names (case-insensitive)
#         limit: Maximum number of results (default: None, returns all)
#     """
#     collection = mongo_client.get_collection(Config.MONGODB_COLLECTIONS["pathway_info"])

#     query = {}
#     if pathway_ids:
#         query["pathway_id"] = {"$in": pathway_ids}
#     if pathway_name_search:
#         query["pathway_name"] = {"$regex": f"^{pathway_name_search}$", "$options": "i"}

#     cursor = collection.find(query, {"_id": 0})
#     if limit:
#         cursor = cursor.limit(limit)
#     results = list(cursor)

#     if not results:
#         return "No pathways found matching the criteria."

#     # Get column names from first result
#     columns = list(results[0].keys()) if results else []

#     return make_table_response(
#         title="KEGG Pathways",
#         columns=columns,
#         rows=results,
#         summary=f"Found {len(results)} pathways"
#     )


@mcp.tool()
def get_stats(stat_type: str = "summary") -> str:
    """
    Get PanKB database statistics including total counts of genomes, genes,
    mutations, and distribution by family/country.

    Args:
        stat_type: Type of statistics - 'summary' for overall counts,
                  'by_family' for family distribution, 'by_country' for geographic distribution
    """
    collection = mongo_client.get_collection(Config.MONGODB_COLLECTIONS["pankb_stats"])

    latest = collection.find_one(sort=[("date", -1)])

    if not latest:
        return "No statistics available."

    date_str = str(latest.get("date", "Unknown"))[:10]

    if stat_type == "summary":
        dimensions = latest.get("pankb_dimensions", "{}")
        if isinstance(dimensions, str):
            dimensions = json.loads(dimensions)

        rows = [{"metric": k, "value": v} for k, v in dimensions.items()]

        return make_table_response(
            title=f"PanKB Database Dimensions ({date_str})",
            columns=["metric", "value"],
            rows=rows,
            summary=f"Overall database statistics as of {date_str}"
        )

    elif stat_type == "by_family":
        genome_counts = latest.get("organism_genome_count", "{}")
        gene_counts = latest.get("organism_gene_count", "{}")

        if isinstance(genome_counts, str):
            genome_counts = json.loads(genome_counts)
        if isinstance(gene_counts, str):
            gene_counts = json.loads(gene_counts)

        rows = [
            {"family": k, "genomes": genome_counts[k], "genes": gene_counts.get(k, 0)}
            for k in sorted(genome_counts.keys(), key=lambda x: genome_counts[x], reverse=True)
        ]

        return make_table_response(
            title=f"PanKB Statistics by Family ({date_str})",
            columns=["family", "genomes", "genes"],
            rows=rows,
            summary=f"Found {len(rows)} families"
        )

    elif stat_type == "by_country":
        country_stats = latest.get("country_strain_count", "{}")
        if isinstance(country_stats, str):
            country_stats = json.loads(country_stats)

        rows = [
            {"country": k.upper(), "strain_count": v}
            for k, v in sorted(country_stats.items(), key=lambda x: x[1], reverse=True)
        ]

        return make_table_response(
            title=f"PanKB Geographic Distribution ({date_str})",
            columns=["country", "strain_count"],
            rows=rows,
            summary=f"Found {len(rows)} countries"
        )

    return f"Unknown stat_type: {stat_type}. Use 'summary', 'by_family', or 'by_country'."
