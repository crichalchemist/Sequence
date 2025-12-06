#!/bin/bash
# Start all Sequence platform services

echo "🚀 Starting Sequence FX Intelligence Platform..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Start directories
cd /home/crichalchemist/Sequence

# 1. Start MQL5 REST API Server
echo -e "${YELLOW}[1/2] Starting MQL5 REST API Server...${NC}"
python mql5/api_server.py &
MQL5_PID=$!
sleep 2
echo -e "${GREEN}✅ MQL5 API Server running on http://localhost:5000${NC}"
echo ""

# 2. Start Streamlit Dashboard
echo -e "${YELLOW}[2/2] Starting Streamlit Matrix Dashboard...${NC}"
streamlit run streamlit_matrix_app.py --server.port 8504 &
STREAMLIT_PID=$!
sleep 3
echo -e "${GREEN}✅ Streamlit Dashboard running on http://localhost:8504${NC}"
echo ""

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║    Sequence Platform is OPERATIONAL                   ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  🔌 MQL5 API Server: http://localhost:5000           ║${NC}"
echo -e "${GREEN}║  📊 Dashboard:       http://localhost:8504           ║${NC}"
echo -e "${GREEN}║  📚 API Docs:        http://localhost:5000/api/v1/docs║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Trap to kill background processes on script exit
trap "kill $MQL5_PID $STREAMLIT_PID" EXIT

# Wait for processes
wait

