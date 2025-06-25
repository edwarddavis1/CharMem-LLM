#!/usr/bin/env python3
"""
Data Curation Script for Harry Potter Character Analysis

This script processes the Harry Potter PDF to find major characters and the pages they appear on.
Characters are searched by full name, first name, and last name variations.
No AI/LLM is used - this is pure text search for speed.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set
import pypdf  # Using pypdf instead of deprecated PyPDF2
import pandas as pd

# Change to project root directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Major Harry Potter characters with their name variations
MAJOR_CHARACTERS = {
    # NOTE remove Harry for now becuase he is in the title
    # "Harry Potter": [
    #     "Harry Potter",
    #     "Harry",
    #     "Potter",
    #     "Harry James Potter",
    #     "Mr. Potter",
    # ],
    "Hermione Granger": [
        "Hermione Granger",
        "Hermione",
        "Granger",
        "Miss Granger",
        "Ms. Granger",
    ],
    "Ron Weasley": [
        "Ron Weasley",
        "Ron",
        "Weasley",
        "Ronald",
        "Ronald Weasley",
        "Mr. Weasley",
    ],
    "Albus Dumbledore": [
        "Albus Dumbledore",
        "Dumbledore",
        "Albus",
        "Professor Dumbledore",
        "Headmaster",
    ],
    "Severus Snape": ["Severus Snape", "Snape", "Severus", "Professor Snape"],
    "Rubeus Hagrid": ["Rubeus Hagrid", "Hagrid", "Rubeus"],
    "Draco Malfoy": ["Draco Malfoy", "Draco", "Malfoy"],
    "Minerva McGonagall": [
        "Minerva McGonagall",
        "McGonagall",
        "Minerva",
        "Professor McGonagall",
    ],
    "Voldemort": [
        "Voldemort",
        "Tom Riddle",
        "You-Know-Who",
        "He-Who-Must-Not-Be-Named",
        "Dark Lord",
    ],
    "Ginny Weasley": ["Ginny Weasley", "Ginny", "Ginevra"],
    "Neville Longbottom": ["Neville Longbottom", "Neville", "Longbottom"],
    "Luna Lovegood": ["Luna Lovegood", "Luna", "Lovegood"],
    "Cho Chang": ["Cho Chang", "Cho"],
    "Cedric Diggory": ["Cedric Diggory", "Cedric", "Diggory"],
    "Petunia Dursley": ["Petunia Dursley", "Petunia", "Aunt Petunia"],
    "Vernon Dursley": ["Vernon Dursley", "Vernon", "Uncle Vernon"],
    "Dudley Dursley": ["Dudley Dursley", "Dudley"],
}


def load_pdf_pages(pdf_path: str) -> List[str]:
    """
    Load PDF pages as text using pypdf.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of page texts
    """
    print(f"Loading PDF: {pdf_path}")

    pages = []
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            print(f"Total pages: {total_pages}")

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    pages.append(page_text)
                    if (page_num + 1) % 50 == 0:
                        print(f"Processed {page_num + 1}/{total_pages} pages...")
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1}: {e}")
                    pages.append("")

    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

    print(f"Successfully loaded {len(pages)} pages")
    return pages


def search_character_in_text(text: str, character_names: List[str]) -> Set[str]:
    """
    Search for character names in text using case-insensitive matching.

    Args:
        text: Text to search in
        character_names: List of name variations to search for

    Returns:
        Set of found character names
    """
    found_names = set()
    text_lower = text.lower()

    for name in character_names:
        # Use word boundaries to avoid partial matches
        pattern = r"\b" + re.escape(name.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found_names.add(name)

    return found_names


def find_characters_in_pages(pages: List[str]) -> Dict[str, Dict[str, List[int]]]:
    """
    Find all character occurrences across all pages.

    Args:
        pages: List of page texts

    Returns:
        Dictionary mapping character names to their page occurrences
    """
    print("\nSearching for characters across all pages...")

    character_pages = {}

    for character_name, name_variations in MAJOR_CHARACTERS.items():
        character_pages[character_name] = {}

        # Initialize tracking for each name variation
        for variation in name_variations:
            character_pages[character_name][variation] = []

        # Search through all pages
        for page_num, page_text in enumerate(pages):
            if not page_text.strip():  # Skip empty pages
                continue

            found_names = search_character_in_text(page_text, name_variations)

            # Record page numbers (1-indexed) for found names
            for found_name in found_names:
                character_pages[character_name][found_name].append(page_num + 1)

        # # Progress update
        # total_pages_found = sum(
        #     len(pages) for pages in character_pages[character_name].values()
        # )
        # if total_pages_found > 0:
        #     print(
        #         f"✓ {character_name}: Found on {len(set().union(*character_pages[character_name].values()))} unique pages"
        #     )
        # else:
        #     print(f"✗ {character_name}: Not found")

    return character_pages


def generate_character_report(character_pages: Dict[str, Dict[str, List[int]]]) -> str:
    """
    Generate a formatted report of character occurrences.

    Args:
        character_pages: Character page mapping data

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("HARRY POTTER CHARACTER OCCURRENCE REPORT")
    report.append("=" * 80)
    report.append("")

    for character_name, name_data in character_pages.items():
        # Get all unique pages for this character
        all_pages = set()
        for pages in name_data.values():
            all_pages.update(pages)

        if not all_pages:
            continue

        sorted_pages = sorted(all_pages)

        report.append(f"CHARACTER: {character_name}")
        report.append("-" * 50)
        report.append(f"Total unique pages: {len(sorted_pages)}")
        report.append(f"First appearance: Page {min(sorted_pages)}")
        report.append(f"Last appearance: Page {max(sorted_pages)}")
        report.append(f"Page range: {min(sorted_pages)}-{max(sorted_pages)}")

        # Show which name variations were found
        report.append("\nName variations found:")
        for variation, pages in name_data.items():
            if pages:
                report.append(
                    f"  • '{variation}': {len(pages)} occurrences on pages {sorted(set(pages))[:10]}{'...' if len(set(pages)) > 10 else ''}"
                )

        # Show page distribution (first 20 pages for brevity)
        if len(sorted_pages) <= 20:
            report.append(f"\nAll pages: {sorted_pages}")
        else:
            report.append(f"\nFirst 20 pages: {sorted_pages[:20]}...")

        report.append("")

    return "\n".join(report)


def create_character_dataframe(
    character_pages: Dict[str, Dict[str, List[int]]],
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from character page data.

    Args:
        character_pages: Character page mapping data

    Returns:
        pandas DataFrame with character analysis
    """
    data = []

    for character_name, name_data in character_pages.items():
        # Get all unique pages for this character
        all_pages = set()
        for pages in name_data.values():
            all_pages.update(pages)

        if not all_pages:
            continue

        sorted_pages = sorted(all_pages)

        # Count total occurrences across all name variations
        total_occurrences = sum(len(pages) for pages in name_data.values())

        # Get name variations that were actually found
        found_variations = [var for var, pages in name_data.items() if pages]

        data.append(
            {
                "Character": character_name,
                "First_Appearance": min(sorted_pages),
                "Last_Appearance": max(sorted_pages),
                "Total_Pages": len(sorted_pages),
                "Total_Occurrences": total_occurrences,
                "Page_Range": f"{min(sorted_pages)}-{max(sorted_pages)}",
                "Found_Variations": ", ".join(found_variations),
                "All_Pages": sorted_pages,
            }
        )

    # Create DataFrame and sort by first appearance
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values("First_Appearance").reset_index(drop=True)

    return df


def save_detailed_csv(
    character_pages: Dict[str, Dict[str, List[int]]], output_path: str
):
    """
    Save detailed character data to CSV format.

    Args:
        character_pages: Character page mapping data
        output_path: Path to save CSV file
    """
    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Character", "Name_Variation", "Page_Number", "Total_Occurrences"]
        )

        for character_name, name_data in character_pages.items():
            for variation, pages in name_data.items():
                if pages:
                    unique_pages = sorted(set(pages))
                    for page in unique_pages:
                        occurrence_count = pages.count(page)
                        writer.writerow(
                            [character_name, variation, page, occurrence_count]
                        )


# MAIN SCRIPT
print("Harry Potter Character Data Curation Script")
print("=" * 50)

# Set up paths
DATA_PATH = "backend/data/books"
book_filename = "Harry-Potter-and-the-Philosophers-Stone.pdf"
pdf_path = os.path.join(DATA_PATH, book_filename)

# Check if PDF exists
if not os.path.exists(pdf_path):
    print(f"Error: PDF file not found at {pdf_path}")
    print("Please ensure the Harry Potter PDF is in the correct location.")

# Load PDF pages
pages = load_pdf_pages(pdf_path)
if not pages:
    print("Failed to load PDF pages. Exiting.")

# Find characters in pages
character_pages = find_characters_in_pages(pages)

# Create and display DataFrame
df = create_character_dataframe(character_pages)
print("\n" + "=" * 80)
print("CHARACTER ANALYSIS DATAFRAME")
print("=" * 80)
df.to_csv(
    "experiments/first_meet_evaluation_data/HP_character_analysis.csv", index=False
)
