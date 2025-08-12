# Job Market Analyzer Pro

A sophisticated, enterprise-grade job market analysis platform with advanced AI-powered features, user authentication, and comprehensive analytics.

## 🚀 Features

### Core Analysis Capabilities
- **Skills Gap Analysis**: AI-powered identification of skill gaps with personalized learning roadmaps
- **Salary Trend Predictions**: Multi-factor salary forecasting with market confidence indicators
- **Industry Comparison Dashboard**: Interactive cross-industry analytics with 10+ industries
- **Resume Optimization**: AI-powered resume analysis with ATS compatibility scoring
- **Geographic Market Analysis**: Cost-of-living adjusted salary comparisons

### Enterprise Features
- **User Authentication**: JWT-based authentication with secure session management
- **Personalized Dashboards**: User-specific analytics based on career goals and preferences
- **Analysis History**: Comprehensive tracking of user analyses with favorites system
- **Advanced Caching**: Performance-optimized caching system for faster analysis
- **Data Persistence**: Robust user profile and preference management

### Interactive Interface
- **Unified Dashboard**: Streamlit-powered comprehensive analytics interface
- **Real-time Visualizations**: Interactive charts, graphs, and geographic maps
- **Export Capabilities**: Multi-format exports (JSON, CSV, HTML, Markdown)
- **Mobile-Responsive**: Optimized for desktop and mobile viewing

## Project Structure

```
Job-Market-Analyzer/
├── src/
│   ├── scrapers/
│   │   ├── indeed_scraper.py      # Indeed job scraper
│   │   ├── linkedin_scraper.py    # LinkedIn job scraper
│   │   └── base_scraper.py        # Base scraper class
│   ├── analyzers/
│   │   ├── salary_analyzer.py     # Salary trend analysis
│   │   ├── skills_analyzer.py     # Skills demand analysis
│   │   ├── location_analyzer.py   # Geographic analysis
│   │   └── company_analyzer.py    # Company insights
│   ├── visualizers/
│   │   ├── charts.py              # Chart generation
│   │   ├── maps.py                # Geographic visualizations
│   │   └── reports.py             # Report generation
│   ├── database/
│   │   ├── db_manager.py          # Database operations
│   │   └── models.py              # Data models
│   ├── dashboard/
│   │   └── app.py                 # Streamlit dashboard
│   └── main.py                    # Main application entry point
├── data/
│   ├── raw/                       # Raw scraped data
│   ├── processed/                 # Cleaned and processed data
│   └── jobs.db                    # SQLite database
├── reports/
│   ├── charts/                    # Generated charts
│   └── pdfs/                      # PDF reports
├── config/
│   ├── settings.yaml              # Configuration settings
│   └── job_sites.yaml             # Job site configurations
├── .env                           # Environment variables
├── .gitignore                     # Git ignore file
└── requirements.txt               # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/liambeckett-ops/Job-Market-Analyzer.git
cd Job-Market-Analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Command Line Interface

**Scrape job data:**
```bash
python src/main.py scrape --site indeed --query "software engineer" --location "New York"
```

**Generate analysis:**
```bash
python src/main.py analyze --type salary --output reports/
```

**Launch dashboard:**
```bash
streamlit run src/dashboard/app.py
```

### Python API

```python
from src.scrapers.indeed_scraper import IndeedScraper
from src.analyzers.salary_analyzer import SalaryAnalyzer

# Scrape job data
scraper = IndeedScraper()
jobs = scraper.scrape_jobs("python developer", "remote")

# Analyze salaries
analyzer = SalaryAnalyzer()
salary_insights = analyzer.analyze_trends(jobs)
```

## Sample Analysis Results

### Salary Trends
- **Average Salary**: $85,000 - $120,000
- **Top Paying Cities**: San Francisco, Seattle, New York
- **Salary Growth**: +12% year-over-year

### In-Demand Skills
1. Python (78% of postings)
2. JavaScript (65% of postings)
3. AWS (52% of postings)
4. Docker (48% of postings)
5. React (45% of postings)

### Top Hiring Companies
1. Google (150 postings)
2. Microsoft (142 postings)
3. Amazon (138 postings)

## Configuration

Edit `config/settings.yaml` to customize:
- Scraping intervals
- Data retention policies
- Visualization preferences
- Export formats

## Ethics & Legal Compliance

This tool respects robots.txt files and implements rate limiting to ensure responsible scraping. Always review and comply with website terms of service.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Additional job board integrations (Glassdoor, ZipRecruiter)
- [ ] Machine learning salary prediction models
- [ ] Real-time job alerts
- [ ] Mobile app version
- [ ] API endpoint for external integrations

## Author

**Liam Beckett Jorgensen**
- GitHub: [@liambeckett-ops](https://github.com/liambeckett-ops)
- LinkedIn: [https://www.linkedin.com/in/liam-jorgensen-191130133/]

---

*Built with Python, BeautifulSoup, Pandas, and CoPilot*
