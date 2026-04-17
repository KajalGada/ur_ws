#include <memory>
#include <vector>
#include <string>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#include "behavior_cloning/action/move_to_joint.hpp"
#include "behavior_cloning/action/execute_plan.hpp"

using MoveToJoint = behavior_cloning::action::MoveToJoint;
using ExecutePlan  = behavior_cloning::action::ExecutePlan;
using GoalHandlePlan    = rclcpp_action::ServerGoalHandle<MoveToJoint>;
using GoalHandleExecute = rclcpp_action::ServerGoalHandle<ExecutePlan>;

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

static const std::string PLANNING_GROUP = "arm";

class MoveItActionServer : public rclcpp::Node
{
public:
    MoveItActionServer()
        : Node("moveit_action_server",
               rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)),
          has_plan_(false),
          move_group_ready_(false)
    {
        using namespace std::placeholders;

        plan_server_ = rclcpp_action::create_server<MoveToJoint>(
            this, "plan_to_joint",
            std::bind(&MoveItActionServer::handle_plan_goal,     this, _1, _2),
            std::bind(&MoveItActionServer::handle_plan_cancel,   this, _1),
            std::bind(&MoveItActionServer::handle_plan_accepted, this, _1)
        );

        execute_server_ = rclcpp_action::create_server<ExecutePlan>(
            this, "execute_plan",
            std::bind(&MoveItActionServer::handle_exec_goal,     this, _1, _2),
            std::bind(&MoveItActionServer::handle_exec_cancel,   this, _1),
            std::bind(&MoveItActionServer::handle_exec_accepted, this, _1)
        );

        // Initialise MoveGroupInterface once in a background thread
        // (shared_from_this() is safe here — the node is fully constructed)
        init_thread_ = std::thread([this]() {
            RCLCPP_INFO(this->get_logger(), "Initialising MoveGroupInterface...");
            move_group_ = std::make_shared<MoveGroupInterface>(shared_from_this(), PLANNING_GROUP);
            move_group_->setPlanningTime(10.0);
            move_group_->setMaxVelocityScalingFactor(0.3);
            move_group_->setMaxAccelerationScalingFactor(0.3);
            {
                std::lock_guard<std::mutex> lock(mg_mutex_);
                move_group_ready_ = true;
            }
            RCLCPP_INFO(this->get_logger(), "MoveIt Action Server ready.");
            RCLCPP_INFO(this->get_logger(), "  plan_to_joint  — plan a trajectory");
            RCLCPP_INFO(this->get_logger(), "  execute_plan   — execute the last stored plan");
        });
    }

    ~MoveItActionServer()
    {
        if (init_thread_.joinable())
            init_thread_.join();
    }

private:
    rclcpp_action::Server<MoveToJoint>::SharedPtr plan_server_;
    rclcpp_action::Server<ExecutePlan>::SharedPtr  execute_server_;

    std::shared_ptr<MoveGroupInterface> move_group_;
    std::mutex mg_mutex_;
    bool move_group_ready_;
    std::thread init_thread_;

    MoveGroupInterface::Plan stored_plan_;
    std::mutex plan_mutex_;
    bool has_plan_;

    // Returns false and logs a warning if MoveGroupInterface isn't ready yet
    bool ensure_ready()
    {
        std::lock_guard<std::mutex> lock(mg_mutex_);
        if (!move_group_ready_)
        {
            RCLCPP_WARN(this->get_logger(), "MoveGroupInterface not ready yet — try again shortly.");
            return false;
        }
        return true;
    }

    // ── plan_to_joint ──────────────────────────────────────────────────────

    rclcpp_action::GoalResponse handle_plan_goal(
        const rclcpp_action::GoalUUID &,
        std::shared_ptr<const MoveToJoint::Goal>)
    {
        RCLCPP_INFO(this->get_logger(), "[plan_to_joint] Goal received");
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_plan_cancel(
        const std::shared_ptr<GoalHandlePlan>)
    {
        RCLCPP_INFO(this->get_logger(), "[plan_to_joint] Cancel requested");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_plan_accepted(const std::shared_ptr<GoalHandlePlan> goal_handle)
    {
        std::thread{std::bind(&MoveItActionServer::execute_plan_action,
                              this, std::placeholders::_1), goal_handle}.detach();
    }

    void execute_plan_action(const std::shared_ptr<GoalHandlePlan> goal_handle)
    {
        auto result = std::make_shared<MoveToJoint::Result>();

        if (!ensure_ready())
        {
            result->success = false;
            goal_handle->succeed(result);
            return;
        }

        auto goal = goal_handle->get_goal();
        std::vector<double> joints(goal->joint_positions.begin(), goal->joint_positions.end());

        {
            std::lock_guard<std::mutex> lock(mg_mutex_);
            move_group_->setJointValueTarget(joints);
        }

        MoveGroupInterface::Plan plan;
        bool success;
        {
            std::lock_guard<std::mutex> lock(mg_mutex_);
            success = (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
        }

        if (!success)
        {
            RCLCPP_WARN(this->get_logger(), "[plan_to_joint] Planning failed");
            result->success = false;
            goal_handle->succeed(result);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(plan_mutex_);
            stored_plan_ = plan;
            has_plan_ = true;
        }

        RCLCPP_INFO(this->get_logger(), "[plan_to_joint] Planning succeeded.");
        result->success = true;
        goal_handle->succeed(result);
    }

    // ── execute_plan ───────────────────────────────────────────────────────

    rclcpp_action::GoalResponse handle_exec_goal(
        const rclcpp_action::GoalUUID &,
        std::shared_ptr<const ExecutePlan::Goal>)
    {
        std::lock_guard<std::mutex> lock(plan_mutex_);
        if (!has_plan_)
        {
            RCLCPP_WARN(this->get_logger(), "[execute_plan] No plan stored. Run plan_to_joint first.");
            return rclcpp_action::GoalResponse::REJECT;
        }
        RCLCPP_INFO(this->get_logger(), "[execute_plan] Goal received");
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_exec_cancel(
        const std::shared_ptr<GoalHandleExecute>)
    {
        RCLCPP_INFO(this->get_logger(), "[execute_plan] Cancel requested");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_exec_accepted(const std::shared_ptr<GoalHandleExecute> goal_handle)
    {
        std::thread{std::bind(&MoveItActionServer::execute_exec_action,
                              this, std::placeholders::_1), goal_handle}.detach();
    }

    void execute_exec_action(const std::shared_ptr<GoalHandleExecute> goal_handle)
    {
        auto result = std::make_shared<ExecutePlan::Result>();

        MoveGroupInterface::Plan plan;
        {
            std::lock_guard<std::mutex> lock(plan_mutex_);
            plan = stored_plan_;
            has_plan_ = false;
        }

        bool success;
        {
            std::lock_guard<std::mutex> lock(mg_mutex_);
            success = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
        }

        result->success = success;

        if (success)
        {
            RCLCPP_INFO(this->get_logger(), "[execute_plan] Execution succeeded.");
            goal_handle->succeed(result);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "[execute_plan] Execution failed.");
            goal_handle->abort(result);
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MoveItActionServer>());
    rclcpp::shutdown();
    return 0;
}
