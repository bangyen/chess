/**
 * Template Dashboard JavaScript
 * 
 * Simple dashboard with navigation, charts, and data loading.
 */
class SimpleDashboard {
    constructor() {
        this.apiBaseUrl = '/api';
        this.charts = {};
        this.init();
    }

    init() {
        this.initNavigation();
        this.initCharts();
        this.bindEvents();
        this.loadData();
    }

    // Navigation
    initNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const views = document.querySelectorAll('.view-container');
        
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const viewName = item.getAttribute('data-view');
                
                // Update active nav item
                navItems.forEach(n => n.classList.remove('active'));
                item.classList.add('active');
                
                // Show selected view
                views.forEach(v => v.classList.add('hidden'));
                document.getElementById(`${viewName}-view`).classList.remove('hidden');
                
                // Update page title
                const titles = {
                    'overview': 'Dashboard Overview',
                    'analytics': 'Analytics',
                    'data': 'Data View'
                };
                document.querySelector('.page-title').textContent = titles[viewName] || 'Dashboard';
            });
        });
    }

    // Charts
    initCharts() {
        // Overview bar chart
        const chart1Ctx = document.getElementById('chart-1')?.getContext('2d');
        if (chart1Ctx) {
            this.charts.chart1 = new Chart(chart1Ctx, {
                type: 'bar',
                data: {
                    labels: ['A', 'B', 'C'],
                    datasets: [{
                        label: 'Dataset 1',
                        data: [0, 0, 0],
                        backgroundColor: ['#E63946', '#1D3557', '#457B9D']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 1.8,
                    plugins: { legend: { display: false } }
                }
            });
        }
        
        // Overview doughnut chart
        const chart2Ctx = document.getElementById('chart-2')?.getContext('2d');
        if (chart2Ctx) {
            this.charts.chart2 = new Chart(chart2Ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Category 1', 'Category 2', 'Category 3'],
                    datasets: [{
                        data: [33.3, 33.3, 33.3],
                        backgroundColor: ['#E63946', '#1D3557', '#457B9D'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 1.8,
                    cutout: '70%',
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                font: { family: 'Space Grotesk', size: 12 },
                                color: '#6E6E73',
                                padding: 16
                            }
                        }
                    }
                }
            });
        }
        
        // Time series chart
        const timeseriesCtx = document.getElementById('timeseries-chart')?.getContext('2d');
        if (timeseriesCtx) {
            this.charts.timeseries = new Chart(timeseriesCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 2,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                font: { family: 'Space Grotesk', size: 12 },
                                color: '#6E6E73',
                                padding: 16
                            }
                        }
                    },
                    scales: {
                        x: { grid: { display: false } },
                        y: { grid: { color: '#E1E4E8' } }
                    }
                }
            });
        }
    }

    // Events
    bindEvents() {
        // Refresh button
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadData());
        }
    }

    // Data loading
    async loadData() {
        this.showLoading('Loading dashboard data...');
        try {
            await Promise.all([
                this.loadOverviewData(),
                this.loadTimeSeriesData(),
                this.loadDetailsData()
            ]);
            this.showSuccess('Dashboard data loaded successfully');
        } catch (error) {
            console.error('Failed to load data:', error);
            this.showError('Failed to load dashboard data');
        } finally {
            this.hideLoading();
        }
    }

    // Overview data
    async loadOverviewData() {
        const response = await fetch(`${this.apiBaseUrl}/overview`);
        const data = await response.json();
        
        // Update metric cards
        const metrics = data.metrics || {};
        Object.keys(metrics).forEach((key, idx) => {
            const elem = document.getElementById(`metric-${idx + 1}`);
            if (elem) elem.textContent = metrics[key];
        });
        
        // Update charts
        if (data.chart_data) {
            if (this.charts.chart1) {
                this.charts.chart1.data.labels = data.chart_data.labels;
                this.charts.chart1.data.datasets[0].data = data.chart_data.values;
                this.charts.chart1.update();
            }
            if (this.charts.chart2) {
                this.charts.chart2.data.labels = data.chart_data.labels;
                this.charts.chart2.data.datasets[0].data = data.chart_data.values;
                this.charts.chart2.update();
            }
        }
    }

    // Time series data
    async loadTimeSeriesData() {
        const response = await fetch(`${this.apiBaseUrl}/timeseries`);
        const data = await response.json();
        this.timeseriesData = data;
        this.updateTimeSeriesChart();
    }

    // Update time series chart
    updateTimeSeriesChart() {
        if (!this.charts.timeseries || !this.timeseriesData) return;
        
        const datasets = ['series_1', 'series_2', 'series_3'].map((key, idx) => {
            const colors = ['#E63946', '#1D3557', '#457B9D'];
            const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            return {
                label: label,
                data: this.timeseriesData[key] || [],
                borderColor: colors[idx % colors.length],
                backgroundColor: colors[idx % colors.length],
                borderWidth: 2,
                pointRadius: 0,
                tension: 0
            };
        });
        
        this.charts.timeseries.data.labels = this.timeseriesData.rounds || [];
        this.charts.timeseries.data.datasets = datasets;
        this.charts.timeseries.update();
    }

    // Details data
    async loadDetailsData() {
        const response = await fetch(`${this.apiBaseUrl}/details`);
        const data = await response.json();
        const tbody = document.getElementById('details-table-body');
        if (!tbody) return;
        
        if (data.items && data.items.length > 0) {
            tbody.innerHTML = data.items.map(item => `
                <tr>
                    <td>${item.name}</td>
                    <td>${item.value}</td>
                    <td>${item.metric?.toFixed(2) || 'N/A'}</td>
                </tr>
            `).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="3">No data available</td></tr>';
        }
    }

    // UI feedback
    showLoading(message = 'Loading...') {
        const loader = document.getElementById('loading-indicator');
        const loaderText = document.getElementById('loading-text');
        if (loader) {
            loader.classList.remove('hidden');
            if (loaderText) {
                loaderText.textContent = message;
            }
        }
    }

    // Hide loading indicator
    hideLoading() {
        const loader = document.getElementById('loading-indicator');
        if (loader) {
            loader.classList.add('hidden');
        }
    }

    // Show success message
    showSuccess(message) {
        this.showToast(message, 'success');
    }

    // Show error message
    showError(message) {
        this.showToast(message, 'error');
    }

    // Show toast notification
    showToast(message, type = 'success') {
        const container = document.getElementById('messages');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = type;
        
        const icon = type === 'success' ? 
            '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M13.5 4.5L6 12L2.5 8.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>' :
            '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 8L8 4M8 12L8 12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="2"/></svg>';
        
        toast.innerHTML = `${icon}<span>${message}</span>`;
        container.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new SimpleDashboard();
});

