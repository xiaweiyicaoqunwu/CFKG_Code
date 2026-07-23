// 全局变量
let graphData = null;
let simulation = null;
let svg = null;
let currentSearchId = 0;
let suggestionAjax = null; // 用于存储当前的建议 AJAX 请求，以便取消

function colorForEntityType(entityType) {
    if (Object.prototype.hasOwnProperty.call(ENTITY_COLORS, entityType)) {
        return ENTITY_COLORS[entityType];
    }

    let hash = 2166136261;
    const typeName = String(entityType);
    for (let index = 0; index < typeName.length; index += 1) {
        hash ^= typeName.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }

    const hue = (hash >>> 0) % 360;
    return `hsl(${hue}, 70%, 50%)`;
}

$(document).ready(function() {
    initializeApplication();
});

function initializeApplication() {
    // 实体类型变化事件
    $('#entity-type').change(function() {
        const entityType = $(this).val();
        if (entityType === 'compound') {
            $('#cas-group').slideDown(300);
        } else {
            $('#cas-group').slideUp(300);
        }
        
        // 当实体类型变化时，如果输入框有内容且长度>=2，重新获取建议
        const prefix = $('#entity-name').val().trim();
        if (prefix.length >= 2) {
            getSuggestions(entityType, prefix, '#suggestions-container', '#entity-name', '#entity-type');
        } else {
            $('#suggestions-container').hide().empty();
        }

    });

    // 搜索输入框输入事件（核心修改：不依赖entity-type）
    // 使用 debounce 处理输入，模拟电商平台的平滑更新
    $('#entity-name').on('input', debounce(function() {
        // 关键：获取当前选择的实体类型（可能是空字符串 ''）
        const entityType = $('#entity-type').val(); 
        const prefix = $(this).val().trim(); 
        
        // 只有当输入长度 >= 2 时才发送请求
        if (prefix.length >= 2) {
            // 无论 entityType 是否为空，都发送请求
            getSuggestions(entityType, prefix, '#suggestions-container', '#entity-name', '#entity-type');
        } else {
            // 输入长度不足时隐藏建议
            $('#suggestions-container').hide().empty();
        }
    }, 300)); // 300ms 防抖时间

    $('#inference-citrus-name').on('input', debounce(function() {
        const prefix = $(this).val().trim();
        if (prefix.length >= 2) {
            getInferenceSuggestions(prefix);
        } else {
            $('#inference-suggestions').hide().empty();
        }
    }, 300));

    // 新增：点击输入框外区域时隐藏建议列表 (提供良好的用户体验)
    $(document).on('click', function(e) {
        // 检查点击事件是否发生在实体搜索的 form-group 内部
        // 确保点击输入框、建议列表和选择框本身不会隐藏建议
        if (!$(e.target).closest('#knowledge-query .search-panel .search-form .form-group').length) {
            $('#suggestions-container').hide().empty();
            $('#inference-suggestions').hide().empty();
        }
    });

    // 搜索按钮点击事件
    $('#search-btn').click(function() {
        performSearch();
    });
    // ...
    // 回车键搜索
    $('#entity-name, #cas-number').keypress(function(e) {
        if (e.which === 13) {
            performSearch();
        }
    });
    $('#inference-citrus-name').keypress(function(e) {
        if (e.which === 13) performInference();
    });

    // 推理按钮点击事件
   $('#infer-btn').click(function () {
        performInference();
    });

    // 关闭详情面板
    $('#close-details').click(function() {
        $('#details-panel').slideUp(300);
    });

// 导航点击事件
    $('.nav-link').click(function(e) {
        const target = $(this).attr('href');
        
        if (target && !target.startsWith('#')) {
            return; 
        }
        // 只有是锚点（#knowledge-query 等）时，才拦截并执行下面的 Tab 切换逻辑
        e.preventDefault();
        
        // 隐藏所有 section
        $('.section').hide();
        
        // 显示目标 section (确保 target 是有效的 ID，如 #qa-system)
        $(target).show();
        
        // 切换激活状态样式
        $('.nav-link').removeClass('active');
        $(this).addClass('active');
    });

    // 初始化图形容器大小
    initializeGraphContainer();
    // 初始化问答系统
    initQASystem();

    // 1. 设置默认实体类型为 citrus
    $('#entity-type').val('citrus');
    // 2. 触发 change 事件以确保 UI 状态正确（如隐藏 CAS 输入框）
    $('#entity-type').trigger('change'); 
    // 3. 填入默认名称
    $('#entity-name').val('Cara Cara Navel Orange');
    // 4. 立即执行搜索
    performSearch();
}

// 修改防抖函数，保留原始上下文
function debounce(func, wait) {
  let timeout;
  return function executedFunction() {
    const context = this; // 保存原始this
    const args = arguments; // 保存参数
    
    const later = () => {
      clearTimeout(timeout);
      func.apply(context, args); // 应用原始上下文
    };
    
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// 初始化图形容器
function initializeGraphContainer() {
    const container = document.getElementById('knowledge-graph');
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;
    svg = d3.select('#knowledge-graph')
        .attr('width', width)
        .attr('height', height);
}


// 获取搜索建议
function getSuggestions(entityType, prefix, containerId, inputId, typeSelectId) {
    // 取消上一个未完成的请求
    if (suggestionAjax) {
        suggestionAjax.abort();
    }
    
    suggestionAjax = $.ajax({
        url: '/suggestions',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            entity_type: entityType, 
            prefix: prefix
        }),
        success: function(response) {
            // 调用显示函数，传入对应的容器ID
            displaySuggestions(response, containerId, inputId, typeSelectId);
        },
        error: function(xhr) {
            if (xhr.statusText !== 'abort') {
                $(containerId).hide().empty();
            }
        },
        complete: function() {
            suggestionAjax = null; 
        }
    });
}

function getInferenceSuggestions(prefix) {
    const citrusRequest = $.ajax({
        url: '/suggestions',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            entity_type: 'citrus',
            prefix: prefix
        })
    });
    const compoundRequest = $.ajax({
        url: '/suggestions',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            entity_type: 'compound',
            prefix: prefix
        })
    });

    $.when(citrusRequest, compoundRequest)
        .done(function(citrusResult, compoundResult) {
            const seen = new Set();
            const merged = citrusResult[0].concat(compoundResult[0]).filter(item => {
                const key = `${item.type}\u0000${item.name}`;
                if (seen.has(key)) return false;
                seen.add(key);
                return true;
            });
            displaySuggestions(merged, '#inference-suggestions', '#inference-citrus-name', null);
        })
        .fail(function() {
            $('#inference-suggestions').hide().empty();
            showMessage('Failed to load suggestions.', 'error');
        });
}

// 显示搜索建议
function displaySuggestions(suggestions, containerId, inputId, typeSelectId) {
    const container = $(containerId);
    container.empty();

    if (suggestions && suggestions.length > 0) {
        suggestions.forEach(item => {
            const itemHtml = `<div class="suggestion-item" 
                                  data-name="${item.name}" 
                                  data-type="${item.type}">
                                <span>${item.name}</span>
                                <span class="suggestion-type">(${item.type})</span>
                              </div>`;
            container.append(itemHtml);
        });
        
        container.show();
        
        // Bind click event for items in this specific container
        container.find('.suggestion-item').on('click', function() {
            const name = $(this).data('name');
            const type = $(this).data('type');
            
            // Fill the specific input
            $(inputId).val(name);
            
            // If there's a type select (Main search only), update it
            if (typeSelectId) {
                $(typeSelectId).val(type).trigger('change');
            }
            
            container.hide().empty();

            if (inputId === '#entity-name') {
                performSearch(); // 如果你想选中后立即搜索，取消注释这行
            }
        });

    } else {
        container.hide().empty();
    }
}

// 执行搜索
// 执行搜索
function performSearch() {
    const entityType = $('#entity-type').val();
    const entityName = $('#entity-name').val();
    const casNumber = $('#cas-number').val();
    const maxPerHopInput = $('#max-per-hop').val().trim(); // 新增：获取输入值
    const hopCount = $('#hop-count').val();  // 获取选择的跳数

    if (!entityType) {
        showMessage('Please select entity type!', 'warning');
        return;
    }

    if (entityType !== 'compound' && !entityName) {
        showMessage('Please input entity name!', 'warning');
        return;
    }

    if (entityType === 'compound' && !entityName && !casNumber) {
        showMessage('Please input entity name or cas number!', 'warning');
        return;
    }

    // 生成新的搜索ID，用于避免旧搜索覆盖新结果
    const searchId = ++currentSearchId;
    
    // 显示加载状态
    showLoading(true);
    
    // 准备请求数据
    const requestData = {
        entity_type: entityType,
        entity_name: entityName,
        cas_number: casNumber
    };
    
    // 新增：处理max_per_hop参数
    if (maxPerHopInput) {
        const parsedValue = parseInt(maxPerHopInput, 10);
        // 只有有效数字（1-500）才添加到请求中
        if (!isNaN(parsedValue) && parsedValue >= 1 && parsedValue <= 500) {
            requestData.max_per_hop = parsedValue;
        }
        // 无效输入时，不添加该参数，后端会使用默认值50
    }

    const parsedHopCount = parseInt(hopCount, 10);
    // 验证跳数范围(1-5)，无效时后端会使用默认值3
    if (!isNaN(parsedHopCount) && parsedHopCount >= 1 && parsedHopCount <= 5) {
        requestData.hop_count = parsedHopCount;
    }
    
    // 发送搜索请求
    $.ajax({
        url: '/search_simple',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(requestData), // 使用动态构建的请求数据
        success: function(response) {
            // 只处理最新的搜索结果
            if (searchId === currentSearchId) {
                if (response.error) {
                    showMessage(response.error, 'error');
                    return;
                }
                
                graphData = response;
                renderKnowledgeGraph();
                showMessage(`Found ${response.counts.entities} entities and ${response.counts.relationships} relations`, 'success');
                // 后端 app.py 在 /search_simple 中返回的 entities 包含 'is_target' 字段
                if (response.entities && response.entities.length > 0) {
                    // 找到被标记为目标（中心）的实体
                    const targetEntity = response.entities.find(entity => entity.is_target);
                    
                    if (targetEntity) {
                        // 调用现有的 showEntityDetails 函数获取并显示完整详情
                        showEntityDetails(targetEntity.id);
                    }
                }
            }
        },
        error: function(xhr) {
            if (searchId === currentSearchId) {
                let errorMessage = '搜索失败';
                try {
                    const errorData = JSON.parse(xhr.responseText);
                    errorMessage = errorData.error || errorMessage;
                } catch (e) {
                    errorMessage = '网络错误，请稍后重试';
                }
                showMessage(errorMessage, 'error');
            }
        },
        complete: function() {
            if (searchId === currentSearchId) {
                showLoading(false);
            }
        }
    });
}

// 显示加载状态
function showLoading(show) {
    const $btn = $('#search-btn');
    if (show) {
        $btn.prop('disabled', true);
        $btn.find('.btn-text').hide();
        $btn.find('.btn-loading').show();
        
        // 显示全局加载指示器
        $('#knowledge-graph').addClass('loading');
    } else {
        $btn.prop('disabled', false);
        $btn.find('.btn-text').show();
        $btn.find('.btn-loading').hide();
        $('#knowledge-graph').removeClass('loading');
    }
}

// 显示消息
function showMessage(message, type = 'info') {
    const messageClass = `message-${type}`;
    const $message = $(`<div class="message ${messageClass}">${message}</div>`);
    
    $('.messages').remove();
    $('.main-content').prepend('<div class="messages"></div>');
    $('.messages').append($message);
    
    setTimeout(() => {
        $message.fadeOut(300, function() {
            $(this).remove();
        });
    }, 3000);
}

// 优化的知识图谱渲染
function renderKnowledgeGraph() {
    // 清除现有图形
    d3.select('#knowledge-graph').selectAll('*').remove();
    
    if (!graphData || (!graphData.entities.length && !graphData.relationships.length)) {
        showEmptyState();
        return;
    }

    const width = document.getElementById('knowledge-graph').clientWidth;
    const height = document.getElementById('knowledge-graph').clientHeight;

    svg = d3.select('#knowledge-graph')
        .attr('width', width)
        .attr('height', height);

    // 创建缩放行为
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);

    const g = svg.append('g');

    // 节点默认半径（如果你将来希望节点半径可变，可把它作为节点属性）
    const NODE_RADIUS = 25;

    // 创建力导向图模拟（优化参数）
    simulation = d3.forceSimulation(graphData.entities)
        .force('link', d3.forceLink(graphData.relationships)
            .id(d => d.id)
            .distance(150)
            .strength(0.5))
        .force('charge', d3.forceManyBody()
            .strength(-400)
            .distanceMin(50)
            .distanceMax(400))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide()
            .radius(NODE_RADIUS + 8)
            .strength(0.8))
        .alphaDecay(0.05)
        .velocityDecay(0.4);

    // LINKS: 使用 g.link-group 包含 line 与 arrowPath（箭头）
    const linkGroup = g.append('g').attr('class', 'links');
    const link = linkGroup.selectAll('.link-group')
        .data(graphData.relationships)
        .enter()
        .append('g')
        .attr('class', 'link-group');

    // 每条关系画一条线（line）并在同组中创建一个 arrowPath（triangle）
    link.append('line')
        .attr('class', 'link')
        .attr('stroke-width', 2);

    // arrow shape: triangle path pointing to +x, positioned by transform on tick
    link.append('path')
        .attr('class', 'arrow-path')
        .attr('d', 'M -8 -5 L 0 0 L -8 5 Z') // tip at (0,0), base behind on -x
        .attr('fill', '#999')
        .attr('pointer-events', 'none'); // 箭头不响应鼠标事件

    // 创建节点（使用分组提高性能）
    const node = g.append('g')
        .selectAll('g')
        .data(graphData.entities)
        .enter()
        .append('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));

    // 添加圆形节点
    node.append('circle')
        .attr('r', NODE_RADIUS)
        .attr('fill', d => colorForEntityType(d.type))
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .attr('class', 'node-circle');

    // 添加文本标签（确保不超出圆圈）
    node.append('text')
        .text(d => truncateToCircle(d.label, NODE_RADIUS))
        .attr('text-anchor', 'middle')
        .attr('dy', 4)
        .attr('font-size', '10px')
        .attr('fill', '#333')
        .attr('font-weight', 'bold')
        .attr('class', 'node-label');

    // 添加事件监听
    addGraphEventListeners(node, link.select('line'));

    // 设置连接引用（使 simulation 的 link force 可以识别对象）
    simulation.force('link').links(graphData.relationships);

    // 更新模拟
    simulation.on('tick', () => {
        updateGraphPositions(link, node, NODE_RADIUS);
    });

    // 自动停止模拟以节省性能（保留一点动能，避免突然停止导致视觉不连贯）
    setTimeout(() => {
        if (simulation) {
            simulation.alphaTarget(0);
            simulation.restart();
        }
    }, 3000);
}

// 更新图形位置（优化版）
function updateGraphPositions(link, node, nodeRadius) {
    // 处理连接线 - 确保边精确连接到节点中心
    link.select('line')
        .attr('x1', d => {
            // 计算源节点到目标节点的角度
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const angle = Math.atan2(dy, dx);
            // 计算源节点边缘的点（而非中心）
            return d.source.x + Math.cos(angle) * nodeRadius;
        })
        .attr('y1', d => {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const angle = Math.atan2(dy, dx);
            return d.source.y + Math.sin(angle) * nodeRadius;
        })
        .attr('x2', d => {
            // 计算目标节点到源节点的角度
            const dx = d.source.x - d.target.x;
            const dy = d.source.y - d.target.y;
            const angle = Math.atan2(dy, dx);
            // 计算目标节点边缘的点
            return d.target.x + Math.cos(angle) * nodeRadius;
        })
        .attr('y2', d => {
            const dx = d.source.x - d.target.x;
            const dy = d.source.y - d.target.y;
            const angle = Math.atan2(dy, dx);
            return d.target.y + Math.sin(angle) * nodeRadius;
        });

    // 处理箭头位置 - 始终指向目标节点
    link.select('.arrow-path')
        .attr('transform', d => {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const angle = Math.atan2(dy, dx) * 180 / Math.PI;
            // 箭头位置在目标节点边缘
            const targetX = d.target.x - Math.cos(angle * Math.PI / 180) * nodeRadius;
            const targetY = d.target.y - Math.sin(angle * Math.PI / 180) * nodeRadius;
            return `translate(${targetX},${targetY}) rotate(${angle})`;
        });

    // 处理节点位置
    node.attr('transform', d => `translate(${d.x},${d.y})`);
}

// 添加图形事件监听器（保持悬停不跳动 + 高亮交互）
function addGraphEventListeners(node, linkLines) {
    // 节点点击事件
    node.on('click', function(event, d) {
        event.stopPropagation();
        showEntityDetails(d.id);
    });

    // 连线点击事件：绑定到 line 元素
    linkLines.on('click', function(event, d) {
        event.stopPropagation();
        showRelationshipDetails(d.id);
    });

    // 节点悬停事件（防止跳动 + 高亮）
    node.on('mouseover', function(event, d) {
        // 固定节点坐标，防止力导向重新计算造成跳动
        d.fx = d.x;
        d.fy = d.y;

        // 放大与高亮（通过半径修改，不使用 CSS transform）
        d3.select(this).select('circle')
            .transition()
            .duration(150)
            .attr('r', 30)
            .attr('stroke', '#000')
            .attr('stroke-width', 3);

        // 高亮相关边（操作 line 的样式）
        d3.selectAll('line.link')
            .transition()
            .duration(150)
            .style('stroke', l => (l.source.id === d.id || l.target.id === d.id) ? '#f66' : '#ccc')
            .style('stroke-width', l => (l.source.id === d.id || l.target.id === d.id) ? 3 : 1);

        // 高亮相关箭头
        d3.selectAll('path.arrow-path')
            .transition()
            .duration(150)
            .attr('fill', l => (l.source.id === d.id || l.target.id === d.id) ? '#f66' : '#999');
    });

    node.on('mouseout', function(event, d) {
        // 释放节点（延迟一下可以避免频繁闪动）
        setTimeout(() => {
            d.fx = null;
            d.fy = null;
        }, 120);

        // 恢复节点样式
        d3.select(this).select('circle')
            .transition()
            .duration(150)
            .attr('r', 25)
            .attr('stroke', '#fff')
            .attr('stroke-width', 2);

        // 恢复所有连线样式
        d3.selectAll('line.link')
            .transition()
            .duration(150)
            .style('stroke', '#999')
            .style('stroke-width', 2);

        d3.selectAll('path.arrow-path')
            .transition()
            .duration(150)
            .attr('fill', '#999');
    });

    // 画布点击事件（清除选择）
    svg.on('click', function() {
        $('#details-panel').slideUp(300);
    });
}


// 拖拽函数
function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;  // 固定x位置
    d.fy = d.y;  // 固定y位置
}

function dragged(event, d) {
    d.fx = event.x;  // 实时更新x位置
    d.fy = event.y;  // 实时更新y位置
    simulation.alpha(0.3).restart();  // 强制模拟更新
}

function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// 文本截断（基于圆半径，保证不会溢出）
function truncateToCircle(text, radius) {
    if (!text) return '';
    // 经验值：每个字符大约占 6px 宽度（font-size 10） -> maxChars = (radius*2 - padding)/6
    const padding = 8;
    const approxCharPx = 6;
    const maxChars = Math.floor((radius * 2 - padding) / approxCharPx);
    if (maxChars < 3) return text.slice(0, Math.max(0, maxChars)) + (text.length > maxChars ? '…' : '');
    return text.length > maxChars ? text.slice(0, maxChars - 1) + '…' : text;
}

// 显示空状态
function showEmptyState() {
    d3.select('#knowledge-graph')
        .append('text')
        .attr('x', '50%')
        .attr('y', '50%')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .text('未找到相关数据')
        .style('font-size', '18px')
        .style('fill', '#999');
}

// 显示实体详情
function showEntityDetails(entityId) {
    showLoading(true);
    
    $.get(`/entity_details/${entityId}`)
        .done(function(response) {
            if (response.error) {
                showMessage(response.error, 'error');
                return;
            }
            displayEntityDetails(response);
        })
        .fail(function() {
            showMessage('获取实体详情失败', 'error');
        })
        .always(function() {
            showLoading(false);
        });
}

// 显示关系详情
function showRelationshipDetails(relId) {
    showLoading(true);
    
    $.get(`/relationship_details/${relId}`)
        .done(function(response) {
            if (response.error) {
                showMessage(response.error, 'error');
                return;
            }
            displayRelationshipDetails(response);
        })
        .fail(function() {
            showMessage('获取关系详情失败', 'error');
        })
        .always(function() {
            showLoading(false);
        });
}

// 显示实体详情内容
function displayEntityDetails(data) {
    const content = $('#details-content');
    content.empty();
    
    content.append(`
        <div class="property-item">
            <div class="property-key">ID:</div>
            <div class="property-value">${data.id}</div>
        </div>
        <div class="property-item">
            <div class="property-key">Label:</div>
            <div class="property-value">${data.type}</div>
        </div>
    `);

    Object.entries(data.properties).forEach(([key, value]) => {
        if (key !== 'id' && key !== 'type') {
            content.append(`
                <div class="property-item">
                    <div class="property-key">${key}:</div>
                    <div class="property-value">${value}</div>
                </div>
            `);
        }
    });

    $('#details-panel').slideDown(300);
}

// 显示关系详情内容
function displayRelationshipDetails(data) {
    const content = $('#details-content');
    content.empty();
    
    content.append(`
        <div class="property-item">
            <div class="property-key">ID:</div>
            <div class="property-value">${data.id}</div>
        </div>
        <div class="property-item">
            <div class="property-key">Relation Label:</div>
            <div class="property-value">${data.type}</div>
        </div>
        <div class="property-item">
            <div class="property-key">Start Node:</div>
            <div class="property-value">${data.start_node.name} (ID: ${data.start_node.id})</div>
        </div>
        <div class="property-item">
            <div class="property-key">End Node:</div>
            <div class="property-value">${data.end_node.name} (ID: ${data.end_node.id})</div>
        </div>
    `);

    Object.entries(data.properties).forEach(([key, value]) => {
        if (!['id', 'type'].includes(key)) {
            content.append(`
                <div class="property-item">
                    <div class="property-key">${key}:</div>
                    <div class="property-value">${value}</div>
                </div>
            `);
        }
    });

    $('#details-panel').slideDown(300);
}

// 窗口大小变化时重新渲染图形
$(window).resize(debounce(function() {
    if (graphData) {
        renderKnowledgeGraph();
    }
}, 250));

// 执行风味推理
function performInference() {
    const citrusName = $('#inference-citrus-name').val().trim();
    
    if (!citrusName) {
        showMessage('Please input citrus!', 'warning');
        return;
    }
    
    const inferenceId = ++currentSearchId;
    showInferenceLoading(true);
    
    // 1. 重置 UI 状态 (先隐藏旧结果)
    $('#existing-grid').empty();
    $('#new-grid').empty();
    $('#existing-section').hide();
    $('#new-section').hide();
    $('#inference-empty').hide();
    // 注意：这里不要隐藏父容器，或者在 success 里必须确保 show()
    
    $.ajax({
        url: '/infer_flavor',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ citrus_name: citrusName }),
        success: function(response) {
            if (inferenceId === currentSearchId) {
                if (response.error) {
                    showMessage(response.error, 'error');
                    return;
                }
                
                // 2. 关键修复：确保父容器是可见的
                $('#inference-results-area').slideDown(300);

                $('#target-citrus-display').text(response.citrus);
                
                const hasExisting = response.existing_results && response.existing_results.length > 0;
                const hasNew = response.new_results && response.new_results.length > 0;

                // 3. 渲染逻辑
                if (!hasExisting && !hasNew) {
                    $('#inference-empty').show();
                } else {
                    // 渲染已存在的结果
                    if (hasExisting) {
                        renderCardGrid(response.existing_results, '#existing-grid');
                        $('#existing-section').fadeIn();
                    }
                    
                    // 渲染新预测的结果
                    if (hasNew) {
                        renderCardGrid(response.new_results, '#new-grid');
                        $('#new-section').fadeIn();
                    }
                }
                
                showMessage('Prediction Complete', 'success');
            }
        },
        error: function(xhr) {
            if (inferenceId === currentSearchId) {
                let msg = 'Inference Failed';
                try { msg = JSON.parse(xhr.responseText).error || msg; } catch(e){}
                showMessage(msg, 'error');
            }
        },
        complete: function() {
            if (inferenceId === currentSearchId) showInferenceLoading(false);
        }
    });
}

// 显示推理加载状态
function showInferenceLoading(show) {
    const $btn = $('#infer-btn');
    if (show) {
        $btn.prop('disabled', true);
        $btn.find('.btn-text').hide();
        $btn.find('.btn-loading').show();
    } else {
        $btn.prop('disabled', false);
        $btn.find('.btn-text').show();
        $btn.find('.btn-loading').hide();
    }
}

// 显示推理结果
function displayInferenceResults(results) {
    // 显示odor结果
    $('#odor-results').html(formatResultList(results.odor));
    
    // 显示taste结果
    $('#taste-results').html(formatResultList(results.taste));
    
    // 显示mouthfeel结果
    $('#mouthfeel-results').html(formatResultList(results.mouthfeel));
    
    // 显示flavor结果
    $('#flavor-results').html(formatResultList(results.flavor));
}

// 格式化结果列表
function formatResultList(items) {
    if (!items || items.length === 0 || (items.length === 1 && items[0] === 'None')) {
        return '<span class="no-result">None</span>';
    }
    
    return '<ul class="result-list">' + 
           items.map(item => `<li>${item}</li>`).join('') + 
           '</ul>';
}


function initQASystem() {
    const $askBtn = $('#ask-btn');
    const $questionInput = $('#question-input');
    const $historyBox = $('#qa-history-box');

    // 辅助函数：滚动到底部
    function scrollToBottom() {
        $historyBox.scrollTop($historyBox.prop("scrollHeight"));
    }

    /**
     * 核心修改：添加消息到历史记录
     * 增加了 Markdown 解析和 HTML 清洗
     */
    function addMessage(role, content) {
        const roleClass = role === 'user' ? 'user-message' : 'gpt-message';
        const avatarRole = role === 'user' ? 'You' : 'BOT';
        const avatarClass = role === 'user' ? 'user-avatar' : 'gpt-avatar';
        
        let processedContent = '';

        if (role === 'gpt') {
            // ----------------------------------------------------
            // 1. 对于 GPT 回复：解析 Markdown + 清洗 HTML
            // ----------------------------------------------------
            if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
                // 将 Markdown 转为 HTML，并过滤潜在的恶意脚本
                processedContent = DOMPurify.sanitize(marked.parse(content));
            } else {
                // 如果库未加载，回退到普通文本显示（防止报错）
                console.warn('marked or DOMPurify not loaded!');
                processedContent = content; 
            }
        } else {
            // ----------------------------------------------------
            // 2. 对于用户输入：进行转义，防止 XSS 攻击
            // ----------------------------------------------------
            // 利用 jQuery 的 text() 方法自动处理特殊字符（如 <, >）
            processedContent = $('<div>').text(content).html();
        }
        
        // 构建 HTML
        const messageHtml = `
            <div class="chat-message ${roleClass}">
                <div class="avatar ${avatarClass}">${avatarRole}</div>
                <div class="message-bubble">${processedContent}</div>
            </div>
        `;
        
        // 插入页面
        $historyBox.append(messageHtml);
        
        // ----------------------------------------------------
        // 3. (可选) 代码高亮：如果引入了 highlight.js
        // ----------------------------------------------------
        if (role === 'gpt' && typeof hljs !== 'undefined') {
            // 查找刚才插入的消息中的代码块并高亮
            $historyBox.find('.message-bubble:last pre code').each(function(i, block) {
                hljs.highlightElement(block);
            });
        }

        scrollToBottom();
    }
    
    // 问答逻辑
    function handleAsk() {
        const question = $questionInput.val().trim();
        if (!question) {
            showMessage('Please input question.', 'warning');
            return;
        }
        
        // 1. 添加用户消息并清空输入框
        addMessage('user', question);
        $questionInput.val('');
        
        // 2. 添加一个临时的 GPT 加载消息 (这里不需要 Markdown 解析)
        const loadingHtml = `<div class="chat-message gpt-message temp-loading-message">
                                 <div class="avatar gpt-avatar">BOT</div>
                                 <div class="message-bubble">
                                    Thinking... <span class="loading-indicator"></span>
                                 </div>
                             </div>`;
        $historyBox.append(loadingHtml);
        scrollToBottom();

        // 3. 禁用输入和按钮
        $askBtn.prop('disabled', true);
        $questionInput.prop('disabled', true);

        // 4. 发送 AJAX 请求
        $.ajax({
            url: '/ask',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ question: question }),
            success: function(response) {
                const answer = response.answer || 'No such information.';
                
                // 5. 移除加载动画
                $historyBox.find('.temp-loading-message').remove();
                
                // 6. 添加真正的 GPT 回答 (addMessage 内部会自动渲染 Markdown)
                addMessage('gpt', answer);
            },
            error: function(xhr) {
                let errorMessage = 'Query failed, please try again.';
                try {
                    const errorData = JSON.parse(xhr.responseText);
                    errorMessage = errorData.error || errorMessage;
                } catch (e) {
                    // 保持默认错误信息
                }
                
                // 移除加载动画
                $historyBox.find('.temp-loading-message').remove();
                
                // 显示错误消息（不使用 addMessage 防止报错信息被当做 Markdown 解析导致样式奇怪）
                const errorHtml = `
                    <div class="chat-message gpt-message">
                        <div class="avatar gpt-avatar">BOT</div>
                        <div class="message-bubble"><span class="error-message">${errorMessage}</span></div>
                    </div>
                `;
                $historyBox.append(errorHtml);
                scrollToBottom();
            },
            complete: function() {
                // 7. 恢复输入和按钮
                $askBtn.prop('disabled', false);
                $questionInput.prop('disabled', false);
                $questionInput.focus(); // 聚焦输入框方便下次提问
            }
        });
    }

    // 绑定事件
    $askBtn.off('click').on('click', handleAsk); // 使用 off 防止重复绑定
    $questionInput.off('keypress').on('keypress', function(e) {
        if (e.which === 13) handleAsk();
    });
    
    // 首次加载时确保滚动到初始消息
    setTimeout(scrollToBottom, 100);
}

function renderCardGrid(results, containerId) {
    const container = $(containerId);
    container.empty();
    
    if (!results || results.length === 0) return;

    results.forEach((item, index) => {
        // 格式化百分比
        const numericScore = Number(item.score);
        const normalizedScore = Number.isFinite(numericScore)
            ? Math.min(1, Math.max(0, numericScore))
            : 0;
        const percent = (normalizedScore * 100).toFixed(1);
        // 获取样式
        const badgeColor = colorForEntityType(item.type);
        // 动画延迟
        const delay = index * 0.05;

        const card = $('<div>', { class: 'prediction-card' })
            .css('animation-delay', `${delay}s`);
        const cardHeader = $('<div>', { class: 'card-header' });
        const typeBadge = $('<span>', { class: 'card-type' })
            .text(String(item.type))
            .css('background', badgeColor);
        const scoreBadge = $('<span>', { class: 'card-score' })
            .text(`${percent}%`);
        cardHeader.append(typeBadge, scoreBadge);

        const cardBody = $('<div>', { class: 'card-body' });
        const flavorName = $('<h4>', { class: 'flavor-name' })
            .text(String(item.name));
        const progressBackground = $('<div>', { class: 'progress-bar-bg' });
        const progressFill = $('<div>', { class: 'progress-bar-fill' }).css({
            width: `${percent}%`,
            background: badgeColor
        });
        progressBackground.append(progressFill);
        cardBody.append(flavorName, progressBackground);
        card.append(cardHeader, cardBody);
        container.append(card);
    });
}
